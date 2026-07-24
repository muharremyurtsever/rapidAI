# Phase 1b step 1 — Batched expert fetch: NEGATIVE result

**Date:** 2026-07-24
**Question:** Does replacing the per-expert Python fetch loop (per-expert `os.pread` + `np.frombuffer` + `mx.array` per tensor part, then `mx.stack`) with a batched fetch (all missing experts read into ONE buffer, ONE conversion per part) reduce per-token orchestration overhead?
**Answer:** No. The shipped per-expert path is already the fastest of four assembly strategies. Change reverted; library unchanged.

## Variants measured

All four keep the math identical (correctness gate re-verified during the experiment: `phase1_correctness_olmoe.json`, `"match": true`).

1. **old** (shipped) — per-expert fetch; cache stores materialized mx rows; bank = `mx.stack` of cached rows.
2. **many** — missing experts' rows preadv'd into one bytearray per part → one `np.frombuffer` + one `mx.array` per part; cache entries are lazy mx slices; bank still `mx.stack`.
3. **banked** — like *many*, plus the bank is assembled as `concat(stack(hits), batched_misses)` so miss slices never enter the call's graph.
4. **npbank** — cache numpy rows; bank assembled with `np.stack`; ONE `mx.array` conversion per part per call.

## Microbench (drift-controlled, within-process, interleaved)

`tools/scripts/run_phase1b_fetch_microbench.py` — real Qwen3-30B layer-0 up_proj
tensors, page-cache warmed, 400 calls x 8 unique experts, 60 MB SLRU
(~69% hit, GPT-OSS-like miss stream). Median ms/call over 3 interleaved trials
(`data/phase1b_fetch_microbench.json`):

| variant | ms/call | vs old |
|---------|--------:|-------:|
| old     | 0.564   | 1.00x  |
| many    | 0.591   | 1.05x slower |
| banked  | 0.634   | 1.12x slower |
| npbank  | 0.806   | 1.43x slower |

cProfile of the *banked* path: `preadv` totals only **0.16 ms/call**; the rest
is MLX op/graph/eval time. Batching the reads therefore has almost nothing to
win, while every batched variant pays at least one extra full copy of the miss
bytes (batch buffer -> conversion -> concat, or np.stack -> conversion). On
GPT-OSS-120B the miss stream is **680 MB/token**, so one extra CPU-side copy is
~30+ ms/token of pure loss.

## Why end-to-end tok/s could not settle this

End-to-end runs during the same afternoon (each a fresh process, one budget per
process, per project rule) drifted monotonically regardless of code version —
Qwen3-30B at 6144 MB: variant 6.27 → 6.12 → old-code 4.84 → variant 4.81 →
variant 4.95 tok/s; identical bytes/token (33.3 MB) and hit rate (95.79%)
throughout. GPT-OSS-120B at 8192 MB measured 0.91 tok/s (variant) vs the 1.04
baseline from the previous day. Host state (likely external-SSD thermal
throttling after ~40+ GB of sustained benchmark reads; `pmset -g therm` shows
no CPU thermal warning) swings tok/s by up to 1.46x, swamping a <10% code
effect. The variant end-to-end records are committed as
`data/phase1_bench_qwen3_30b_p1b_6144.json` and
`data/phase1_bench_gptoss120b_p1b_8192.json`; canonical baselines remain
`phase1_bench_qwen3_30b_o1slru.json` (7.08 tok/s) and
`phase1_bench_gptoss120b_8192.json` (1.04 tok/s).

**Methodology rule added:** cross-day (or cross-hour) end-to-end tok/s numbers
are not comparable on this rig; code-level A/B claims need drift-controlled
within-process microbenches or same-session interleaved pairs.

## Conclusions

- The Python bank-assembly floor is ~0.56 ms/call (~0.16 ms preads + ~0.40 ms
  MLX graph/eval) ≈ 80 ms/token on Qwen3-30B (144 switch calls/token). Python
  restructuring cannot push below this — every variant tried is worse.
- The remaining orchestration overhead is MLX op/graph cost, not syscalls.
  Killing it requires either (a) a **persistent resident mx expert bank** per
  store with scatter-updates on miss and `gather_qmm` row indices (no per-call
  stack at all), or (b) the planned **C++/Metal port** of fetch → bank →
  gather_qmm.
- Negative result committed per project rule; library code unchanged.
