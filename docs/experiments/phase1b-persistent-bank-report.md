# Phase 1b step 2 — Persistent resident expert bank: LOSES the pre-registered gate (1.124x < 1.15x)

**Date:** 2026-07-24
**Question:** Does replacing the per-call compact bank (SLRU of mx rows + `mx.stack` + remapped `gather_qmm` indices) with a PERSISTENT preallocated expert bank per store (slot LRU, batched scatter-write of misses, `gather_qmm` with slot indices — no per-call stack) raise end-to-end tok/s by the pre-registered >= 1.15x?
**Answer:** No — it is consistently faster, but only 1.124x end-to-end (and 1.02x at microbench level). Below the gate; library code reverted per the decision rule. The full implementation is preserved inline in `tools/scripts/run_phase1b_bank_microbench.py`.

## Design measured

Per DiskExpertStore (layer x projection): preallocated `mx.zeros((capacity_slots, *row_shape))` per tensor part (weight/scales/biases[/bias]); O(1) slot-granularity LRU (expert -> slot, free list, evict-LRU-reuse-slot); on miss, rows are pread per expert (batched reads stay dead per step 1) and written with ONE batched `bank[slots] = rows` scatter per part; `gather_qmm` runs on the persistent bank with slot indices. Calls wider than the slot capacity (prefill batches) fall back to a one-shot compact bank without touching the LRU. Capacity = cache-byte budget split evenly across stores, in whole rows, capped at n_experts. Verified before building: MLX 0.32 `__setitem__` scatter works on uint32/bf16 banks, and `gather_qmm` cost does not scale with bank row count (8 vs 128 rows: 1.07 -> 0.44 ms, i.e. warmup-dominated, no penalty).

Correctness chain, all green while the persistent path was installed:

- 24/24 unit tests, including: slot reuse/eviction order, within-call MRU protection, math equality vs reference `gather_qmm` through forced evictions (capacity 2 < E=4), slot-reuse-without-refetch, oversize-call fallback equality.
- `run_phase1_correctness.py` with `RAPIDAI_BANK=persistent`: **"match": true** (OLMoE greedy tokens byte-identical).
- After the revert, shipped code re-verified: 19/19 tests, gate **"match": true**.

## Microbench (interleaved, within-process, full call path incl. gather_qmm)

`tools/scripts/run_phase1b_bank_microbench.py` — real Qwen3-30B layer-0
up_proj, page-cache warmed, 400 calls x 8 unique experts, 60 MB budget
(~69% hit). Unlike the step-1 fetch microbench, both variants run assembly +
`gather_qmm` + eval, since the persistent bank changes the gather input too.
Median ms/call over 3 interleaved trials (`data/phase1b_bank_microbench.json`):

| variant | ms/call | vs percall |
|---------|--------:|-----------:|
| percall (shipped) | 0.561-0.572 | 1.00x |
| persistent | 0.547-0.560 | **1.02x faster** |

(Toy-layer trial 0 is always slower for both — process warmup; medians shown.)
The per-call `mx.stack` of 8 rows is simply cheap; the persistent path trades
it for slot bookkeeping + an occasional scatter and nets almost nothing at
single-layer granularity.

## End-to-end (interleaved, fresh process per run, alternating variants)

Qwen3-30B-A3B 3-bit @ 6144 MB, 256 decode tokens, runs executed
percall,persistent,percall,persistent,percall,persistent back-to-back
(`data/phase1_bench_qwen3_30b_bankAB_*.json`):

| run | percall tok/s | persistent tok/s |
|----:|--------------:|-----------------:|
| r0 | 5.092 | 5.721 |
| r1 | 5.073 | 5.726 |
| r2 | 5.090 | 5.700 |
| **median** | **5.090** | **5.721** |

Ratio: **1.124x** (gate: >= 1.15x) → **LOSE**. Both variants were rock-stable
within the session (spread < 0.6%), confirming the interleaved methodology
controls the 1.46x host drift that invalidated cross-session comparisons.
Persistent's hit rate is slightly lower (95.48% vs 95.79%, 35.7 vs 33.2
MB/token) because the budget is split evenly across stores instead of shared
globally by the SLRU — it wins on time despite reading more bytes.

GPT-OSS-120B runs were pre-registered as conditional on a win → not run.

## Why the e2e gain (12.4%) exceeds the microbench gain (2%)

At 144 switch calls/token the persistent path saves graph-construction and
allocator churn that the single-layer microbench cannot see (no cross-layer
memory pressure, no 48-layer graph). But the dominant per-call cost — MLX op
dispatch, index conversion, eval sync — is untouched, which is why the effect
saturates near 12%.

## Conclusions

- Pre-registered rule applied: 1.124x < 1.15x → library reverted to the
  shipped per-call path. Implementation preserved (inline) in the microbench
  script; this report + committed data keep the numbers.
- The result sharpens the step-1 conclusion: Python/MLX orchestration
  restructuring is now measured twice as exhausted. A ~12% ceiling exists via
  the persistent bank, but it does not justify a second call path with
  fixed-split cache behavior and prefill fallbacks.
- **Next lever is the C++/Metal port of fetch -> bank -> gather_qmm** (Phase
  1b as originally scoped, ceiling >10 tok/s on 30B). If the port keeps a
  Python-visible bank, the persistent-slot design (this experiment) is the
  right layout for it — slot LRU + scatter maps 1:1 onto a Metal residency
  buffer, and its correctness suite already exists in this script.
