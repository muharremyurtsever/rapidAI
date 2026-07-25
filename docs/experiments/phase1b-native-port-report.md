# Phase 1b step 3 — Native (C++/Metal) lazy-fetch port: LOSES the pre-registered gate (1.136x < 1.5x)

**Date:** 2026-07-25
**Question:** Does moving the expert fetch out of Python and out of graph-build
time — a C++ MLX extension whose primitive fetches experts at graph-EVAL time
inside the Metal pipeline — raise end-to-end decode by the pre-registered
>= 1.5x on Qwen3-30B @ 6144 MB?
**Answer:** No — 1.136x (5.802 vs 5.109 tok/s interleaved medians). The
library keeps the shipped Python per-call path. The extension, its tests and
the standalone installer are kept as a reproducible experiment
(`tools/native/`, `rapidai_tools/native_moe.py`,
`tools/scripts/run_phase1b_native_bench.py`).

## 1. Investigation: where the 0.55 ms/call actually goes

`tools/scripts/run_phase1b_native_investigation.py`
(`data/phase1b_native_investigation.json`), real Qwen3-30B layer-0 up_proj,
page-cache warm, 100% cache hits (pure orchestration):

| phase (median/call) | ms |
|---|---:|
| np.array(indices) + np.unique | 0.013 |
| SLRU fetch (8 hits) | 0.004 |
| mx.stack graph build | 0.009 |
| gather_qmm graph build | 0.005 |
| **mx.eval** | **0.630** |

- Python/dict/graph-build work is ~0.03 ms — irrelevant. **~95% of the cost
  is the eval**, i.e. the per-call sync + kernel launch pattern.
- Eval-per-call vs one chained eval over 400 calls: 0.448 vs **0.170
  ms/call** → ~0.28 ms/call is pure command-buffer/sync overhead.
- Ceiling: persistent resident bank + PRECOMPUTED slot indices, 144 chained
  gather_qmm in ONE graph: **6.3 ms/token = 0.044 ms/call** (12.7x below the
  Python floor). This is what a native port could reach *if* fetching imposed
  no synchronization at all.
- `mx.compile`: **incompatible** with streamed fetching, verified:
  "[eval] Attempting to eval an array during function transformations like
  compile or vmap is not allowed" — fetch needs indices DATA at graph-build
  time, tracers have none.

Conclusion of the investigation: the bottleneck is (d) the sync pattern —
`np.array(indices)` forces a partial eval per MoE layer, splitting each token
into ~48 sequential graph segments. Chosen approach: **(i) MLX C++ extension**
(nanobind, against the wheel's shipped headers/libmlx) that defers the fetch
to graph-eval time so the token stays one lazy graph.

## 2. What was built

`tools/native/rapidai_bank.cpp` — `_rapidai_bank` extension (nanobind,
NB_DOMAIN mlx, links the wheel's `libmlx.dylib`):

- `NativeExpertStore`: per-LAYER persistent expert bank. Per tensor part a
  preallocated unified-memory buffer (`mlx::core::allocator::malloc`), O(1)
  slot LRU, `pread` of missing expert rows DIRECTLY from the safetensors
  files into the bank slot (zero staging copies — the Python path pays
  np.frombuffer -> mx.array -> stack per call). Supports stacked and
  per-expert (sharded) layouts, mxfp4/affine parts, GPT-OSS additive bias.
- `BankFetch` primitive (`bank_fetch(indices, store) -> [slot_idx, *banks]`):
  runs on the model's own stream. At eval it encodes
  `encodeSignalEvent(v_gpu)` right after the producer of `indices`, registers
  an `MTLSharedEventListener` handler that runs LRU+pread and sets `v_cpu`,
  and encodes `encodeWait(v_cpu)` so the downstream `mx.gather_qmm` is gated.
  No Python sync, no MLX cross-stream machinery; dependencies telescope in a
  single Metal queue.
- The three projections of a layer share the routing indices → ONE store and
  ONE fetch per layer (48 round-trips/token instead of 144 on Qwen3-30B).

Correctness chain (all green, `data/phase1b_native_correctness.json`):
27/27 unit tests (reference-vs-native math incl. forced evictions with
capacity 2 < E, slot-reuse-without-refetch, per-expert layout, linear bias,
oversize-call prefill fallback, chained single-eval laziness, shared
layer-bank fetch) and the OLMoE greedy gate **"match": true** with the native
installer; shipped-path gate re-verified after the revert ("match": true).

### Two failure modes found on the way (worth recording)

1. **MLX cross-stream stall.** The first design ran the primitive on MLX's
   CPU stream with the framework's cross-stream fences. It worked under
   per-token blocking evals but reliably died with a GPU watchdog timeout
   ("Command buffer execution failed: GPU Timeout") on the in-flight token
   that mlx_lm's async pipelining abandons at the end of generation — the
   CPU<->GPU fence ping-pong of a fully-async graph stalls inside large
   command buffers. Single-stream signal/wait avoids the machinery entirely.
2. **GIL deadlock via Metal listener handlers.** `mx.synchronize`'s binding
   does NOT release the GIL. A nanobind-cast `shared_ptr` captured in a Metal
   listener handler needs the GIL in its destructor (Python refcount) — so
   handler destruction blocked the serial listener queue while the main
   thread held the GIL inside `mx.synchronize`, and the GPU watchdog killed
   the stalled buffers ~10 s later. Fix: the store is owned by a plain C++
   `shared_ptr` behind a thin Python handle; nothing on the listener queue
   ever touches Python state.

## 3. End-to-end (pre-registered protocol: fresh process per run, interleaved, medians)

Qwen3-30B-A3B 3-bit @ 6144 MB, 256 decode tokens, runs alternated
percall,native x3 (`data/phase1_bench_qwen3_30b_nativeAB_*.json`):

| run | percall tok/s | native tok/s |
|----:|--------------:|-------------:|
| r0 | 5.066 | 5.797 |
| r1 | 5.109 | 5.802 |
| r2 | 5.161 | 5.822 |
| **median** | **5.109** | **5.802** |

Ratio **1.136x** (gate >= 1.5x) → **LOSE**. Within-variant spread < 1%
(interleaved methodology holds). GPT-OSS-120B was pre-registered as
conditional on a win → not run. Library integration reverted per the decision
rule.

Secondary observations:

- Native reads MORE: 46.9 vs 33.2 MB/token (hit 94.07% vs 95.79%) because the
  budget is split evenly across layers instead of shared globally by the
  SLRU — same fixed-split penalty as the Phase 1b step-2 persistent bank; it
  still wins on time despite the extra bytes.
- Isolated floor of the native path (48 chained fetch+gather, all hits, one
  eval): **72 ms/token = 1.50 ms/call** — *worse per call than the shipped
  path's 0.56 ms*. The Metal signal → listener dispatch → setSignaledValue →
  GPU-resume round-trip costs ~1.4 ms, i.e. it replaces Python's per-layer
  sync with a comparable hardware/scheduler latency.

## 4. Why the 12.7x microbench ceiling did not survive contact

The investigation's 6.3 ms/token floor assumed slot indices exist BEFORE the
graph runs. With data-dependent fetching, a CPU round-trip per MoE layer is
unavoidable no matter where the code lives: the router's output must reach
the CPU (to read the disk) and the CPU's slot table must reach the GPU. The
Python path pays it as `np.array(indices)` (~0.55 ms incl. eval overhead);
the native path pays it as a Metal shared-event listener round-trip
(~1.5 ms isolated, cheaper in-context because it overlaps with model
compute). Removing Python only re-labels the latency.

**The real lever is removing the data dependency from the critical path:**
predict expert sets ahead of time (Bet B, measured overlap 5.5-7.4x random at
t+1..t+8) and prefetch asynchronously so the fetch is complete before the
gather needs it. The persistent-bank layout + native pread-into-slot
machinery built here is exactly the substrate a Bet B prefetcher needs — the
extension is kept for that reason.

## 5. Conclusions

- Pre-registered rule applied: 1.136x < 1.5x → library stays on the shipped
  Python per-call path; native path preserved as a standalone experiment
  behind `install_streaming_native` (never imported by the library).
- Phase 1b as originally scoped ("C++ port of fetch->bank->gather") is now
  MEASURED DEAD as a pure orchestration play: Python glue was never the
  bottleneck (0.03 ms/call); per-layer synchronization is, and it is
  intrinsic to data-dependent streaming.
- Next step: Bet B t+k expert prediction + async prefetch into the native
  slot bank, which converts the per-layer round-trip from blocking to
  overlapped, THEN re-approach the >10 tok/s ceiling.
