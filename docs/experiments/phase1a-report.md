# Phase 1a Report — MLX Streamed-MoE Prototype

**Date:** 2026-07-23
**Machine:** MacBook M3 Pro 18 GB (reference)
**Both pre-registered gates: PASSED.**

## What was built

A drop-in streaming layer for MLX MoE models (`tools/rapidai_tools/`):

- `STReader` — safetensors random access via `os.pread` (the fast path proven in exp 0.2; no mmap).
- `SLRUCache` — segmented LRU with byte accounting (probationary/protected).
- `DiskExpertStore` + `StreamedQuantizedSwitchLinear` — replaces `QuantizedSwitchLinear`: fetches only the experts each token routes to, assembles a compact bank, runs the same `mx.gather_qmm`. Supports both on-disk layouts (Qwen3 stacked tensors, OLMoE per-expert tensors).
- `install_streaming(model, model_dir, cache_bytes)` — one call converts any loaded mlx-lm MoE model to disk-streamed experts.

19/19 unit tests pass. All code, data, plots committed.

## Gate 1 — Correctness (OLMoE, 64 greedy tokens)

Streamed model output is **byte-identical** to the fully-resident model (`"match": true`). Same kernels, same math — streaming changes where weights live, not what the model computes.

## Gate 2 — Viability (Qwen3-30B-A3B 3-bit, 256 decode tokens per run)

| expert-cache budget | tok/s | bytes read / token | decode hit rate |
|---|---|---|---|
| 2 GB (~17% of experts) | **1.83** | 310 MB | 60.7% |
| 4 GB (~33%) | 1.71 | 100 MB | 87.3% |
| 6 GB (~50%) | **2.67** | 33 MB | 95.8% |

Gate required ≥ 1 tok/s at ≤ 6 GB: passed at every budget, including 2 GB. **A 30B-parameter model decoding with ~3 GB total weight residency** (1 GB non-expert + 2 GB expert cache) is the thesis working in prototype form.

OLMoE sweep (same harness, 256 tokens):

| budget | tok/s | bytes/token | hit rate |
|---|---|---|---|
| 256 MB | 6.9 | 451 MB | 0% |
| 512 MB | 7.1 | 373 MB | 17% |
| 1 GB | 9.0 | 199 MB | 56% |
| 2 GB | 11.8 | 58 MB | 87% |
| 3.5 GB (≈all) | 20.6 | 4.5 MB | 99% |

Caveat: OLMoE's 3.5 GB file fits in the OS page cache, so its tok/s is optimistic about disk; the Qwen3 runs (13 GB file) are the honest disk numbers.

## What the numbers say about the equation

- **M_miss is better than projected.** Phase 0 assumed 25–50% miss; measured misses at 33%/50% cache fractions are 13%/4%. The multi-token working set concentrates harder than single-lag overlap suggested (plain SLRU, no predictor yet — Bet B's predictor and Bet E's amortizer are still unplayed cards).
- **The bottleneck moved, as planned.** At 4–6 GB budgets, I/O per token (33–100 MB ≈ 8–24 ms at measured SSD speed) is a small fraction of the ~370–590 ms token time. The remainder is Python orchestration: per-call `np.unique`, per-expert cache dict traffic, compact-bank stacking, and mx graph sync per MoE block × 48 layers. **This overhead is exactly what Phase 1b's C++/Metal port removes** — the I/O math it will inherit is now proven.
- Projection sanity check: at 6 GB the pure-I/O floor would be ~30+ tok/s; even halved by compute, the Phase-6 GPT-OSS-120B target (≥ 3 tok/s) remains inside the envelope, now with a measured M_miss curve behind it instead of an assumption.

## Deviations

- Phase 1a is Python/MLX, not the spec §5 C++ engine — deliberate (one-day iteration cadence); spec architecture unchanged as the Phase 1b target.
- Bench prompt fixed (TCP congestion control, 256 tokens); domain sensitivity of hit rates deferred to Phase 2 benchmarking.

## Next (Phase 1b / Phase 2 candidates)

1. **Phase 2 (Bet E)** — draft model + speculative loop over the streamed target: divides the remaining misses by k≈2.5 and amortizes Python overhead across verified batches. Likely the biggest cheap win.
2. **Phase 1b** — port hot path (fetch + compact-bank + gather) to C++/Metal once E's shape is known.
3. Async prefetch thread (issue next-layer fetches during attention compute) — hides most of the remaining I/O latency.
