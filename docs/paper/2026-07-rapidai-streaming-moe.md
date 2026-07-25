# Streaming Mixture-of-Experts Inference on Consumer Apple Silicon: A Measured Map of What Works and What Doesn't

**Draft preprint — not yet submitted. 2026-07-25.**
Muharrem Yurtsever. Reference hardware: MacBook M3 Pro, 18 GB unified memory.

---

## Abstract

We ask whether a large Mixture-of-Experts (MoE) language model must fit in RAM to run at usable speed on consumer hardware. We built a streaming inference engine for Apple Silicon that keeps a model's attention/norm/embedding tier resident and fetches MoE expert weights from SSD on demand through a segmented-LRU cache, and we evaluated it under pre-registered live/kill gates. Three positive results: (1) GPT-OSS-120B (117B parameters, 5.1B active, ~63 GB on disk) decodes coherently at **1.04 tok/s on an 18 GB machine** — a model 3.5× larger than total RAM; (2) Qwen3-30B-A3B decodes at **up to 7.08 tok/s with a 6 GB expert cache** (~3 GB total weight residency); (3) we contribute the first multi-token expert-routing predictability measurements on modern decoder MoEs, finding routing-overlap that does *not* decay across an 8-token horizon. Equally important are five pre-registered *negative* results that map the design space precisely: shared-base + low-rank expert decomposition (D²-MoE-style) does not transfer to fine-grained MoEs; speculative decoding *increases* per-token disk traffic for expert streaming; and three successive engine optimizations (batched fetch, persistent expert bank, a native C++/Metal fetch primitive) each fail their speed gate because the bottleneck is a per-layer CPU↔GPU round-trip *intrinsic to data-dependent fetch*. A lightweight expert predictor cannot remove that round-trip: predictive signal exists and grows with model scale (+9.66 percentage points over persistence at 120B vs ~0 at 1–3B active) but is far too weak (23% miss coverage) to close the speed gap. We conclude that the ≥3 tok/s target is unreachable for streaming MoE inference on this hardware class, and we release the engine, all measurements, and all negative results so others need not re-derive this map.

## 1. Thesis and the four-factor equation

Conventional wisdom caps the largest runnable model at a machine's RAM. We hypothesized this cap is an artifact of eager weight loading, not physics: *a model does not need to be resident in RAM to think — only the part that is thinking right now does.* The physical quantity that actually governs feasibility is disk-to-memory traffic per generated token, which decomposes as

```
T_disk/token = (P_active × M_miss × B_bytes-per-param) / k_accept
```

- `P_active` — parameters touched per token (MoE routing selects a sparse subset);
- `M_miss` — fraction of touched parameters not resident in a RAM cache;
- `B_bytes-per-param` — storage density (quantization, entropy coding);
- `k_accept` — tokens produced per full-model invocation (speculative decoding).

Each factor has been attacked in isolation in prior work, on NVIDIA/PCIe hardware, in mutually unaware papers. We tested whether composing them on Apple Silicon unified memory clears a usable-speed bar. **Section 6 shows the equation itself was wrong in one term (§4.2) and that a term the equation does not model — a per-layer synchronization floor — dominates.**

## 2. Method: pre-registered gates

Every experiment fixed a live/kill threshold *before* running, and a bet that missed its gate was dropped without revision. This discipline is the paper's methodological spine: it is why the negative results are trustworthy rather than post-hoc rationalizations, and it repeatedly saved weeks by killing dead bets in days. All benchmarks are process-isolated (one model load per process — in-process sweeps corrupt tok/s) and, for code-level A/B claims, interleaved within a session (this rig exhibits up to 1.46× end-to-end tok/s thermal drift across hours, which invalidates cross-session comparison).

## 3. Reference-hardware measurements (Phase 0)

**I/O (exp 0.2).** On the internal NVMe, cache purged per measurement: raw sequential read 3.1–4.2 GB/s; **`pread` into a shared Metal buffer with the GPU consuming concurrently sustains 4.16 GB/s** (full SSD speed); the `mmap` page-fault path caps at **1.4–1.9 GB/s regardless of prefetch strategy** (`madvise(WILLNEED)`, `fcntl(F_RDADVISE)`, or 8 threads); 16 KB random reads collapse to 0.13 GB/s. **Finding: on Apple Silicon, bulk weight streaming should use `pread` into a buffer pool, not `mmap` faulting — inverting the llama.cpp-era default — and expert blocks must be ≥2 MB contiguous.**

**Routing locality (exp 0.1).** Instrumenting every MoE gate over 25,625 decode tokens on Qwen3-30B-A3B: the expert set of token *t* overlaps token *t+1*'s by **7.36× the random baseline** (46% absolute), and the signal **plateaus at ~5.5–6× through t+8** rather than decaying. OLMoE-1B-7B shows the same shape (3.3× → ~2.5×). To our knowledge these are the first multi-token-lookahead routing-decay measurements on modern decoder MoEs. Implication: a cache exploiting temporal locality should reach high hit rates (confirmed in §5).

**Draft acceptance (exp 0.4).** A Qwen3-0.6B draft against a Qwen3-30B target yields mean accepted run length k=2.54 — healthy, but §4.2 shows this does not help expert streaming.

## 4. Negative results that shape the design

### 4.1 Shared-base + low-rank expert deltas do not transfer (Bet A, killed)

D²-MoE (ICML 2025) compresses coarse-expert MoEs (Mixtral-class) as a shared base plus low-rank per-expert deltas. On Qwen3-30B-A3B (128 fine-grained experts), across 15 (layer × projection) combinations, a rank-25% SVD of the mean-subtracted deltas captures only **0.564** of Frobenius energy (gate: ≥0.70), and delta-energy ≈ raw-energy to within 0.003 — the fine-grained experts share essentially **no** common base. Streaming a shared base plus deltas buys nothing here. Modern MoE training drives specialization that defeats the decomposition.

### 4.2 Speculative decoding hurts expert streaming (Bet E, killed) — the equation is wrong

The four-factor equation assumes `k_accept` divides disk traffic. It does not, for routed experts: verifying k drafted tokens in one pass requires the **union** of k tokens' expert sets simultaneously (routing overlap is only ~46%, so the union is ~3–4× a single token's set) under one cache. Measured on the streamed 30B at 4 GB: **0.48× the tok/s and 1.72× the bytes** of plain decoding, despite 59% draft acceptance. Corrected model:

```
T_disk/token = P_active(routed) × M_miss × B/param   +   P_shared × M_miss_shared × B / k_accept
```

`k_accept` survives only on the small dense/shared tier — which is resident anyway. **Bet E is dead for expert streaming.**

## 5. The streaming engine and its positive results (Phase 1a)

We replace each MoE block's expert bank with a disk-backed module: on each call it reads the routed experts' quantized rows via `pread` through a segmented-LRU cache and runs the identical `gather_qmm`. Correctness is exact — streamed OLMoE produces token-identical greedy output to the fully-resident model.

**Qwen3-30B-A3B (3-bit), 256 decode tokens, process-isolated, O(1) cache:**

| expert cache | tok/s | bytes/token | hit rate |
|---|---|---|---|
| 2 GB | 6.75 | 310 MB | 61% |
| 4 GB | 4.52 | 100 MB | 87% |
| 6 GB | **7.08** | 33 MB | 96% |

A 30B model decodes at usable speed with ~3 GB of weights resident. Measured `M_miss` (4–13% at 33–50% cache) beats our Phase-0 projection — the multi-token working set concentrates harder than single-lag overlap implied. (Profiling note: an O(n) byte-sum in our own cache was 59% of decode time; O(1) counters gave 1.74×. Always keep cache bookkeeping O(1).)

**GPT-OSS-120B (MXFP4-Q4), 8 GB cache: 1.04 tok/s, 64% hit, 680 MB/token — coherent output on a machine with 18 GB of RAM for a 63 GB model.** The thesis holds at the "it runs" level for the flagship model.

## 6. Why usable speed is unreachable here (Phase 1b + Bet B)

At high hit rate the bottleneck is no longer disk. We attacked the per-call overhead three ways, each with a pre-registered speed gate on Qwen3-30B @6 GB:

| optimization | interleaved-median result | gate | verdict |
|---|---|---|---|
| batched fetch (group disk reads) | 1.05–1.43× **slower** | — | negative (preadv is 0.16 ms; batching adds copies) |
| persistent resident expert bank | 1.124× | ≥1.15× | miss |
| native C++/Metal lazy-fetch primitive | 1.136× | ≥1.50× | miss |

The native port — a correct nanobind MLX extension with a per-layer persistent bank, O(1) slot LRU, `pread` straight into unified-memory slots, and a single-stream Metal signal→listener-fetch→wait primitive (whole token = one lazy graph, zero Python sync) — isolates the cause: **data-dependent fetch forces one CPU↔GPU round-trip per MoE layer in any language**, because the experts to gather are unknown until routing is computed. The Metal listener round-trip (~1.5 ms/call) merely re-labels Python's 0.56 ms sync. The graph-only floor with precomputed slots is 6.3 ms/token (a 12.7× ceiling) but is reachable only by removing the dependency from the critical path.

The one lever that could do so is prediction (Bet B): forecast a future token's experts from the current hidden state and prefetch asynchronously. We measured predictability directly (gate: trained predictor beats a persistence baseline by ≥15 pp and reaches ≥60% recall@k on 120B):

| model (active params) | trained − persistence @ t+1 | trained miss-coverage |
|---|---|---|
| OLMoE (1B) | −1.1 pp | ~0% |
| Qwen3-30B (3B) | −1.4 pp | 15% |
| GPT-OSS-120B (5B) | **+9.66 pp** | **23%** |

**New finding: routing predictability is scale-dependent** — absent at 1–3B active, real at 5B. But it fails the gate, and the arithmetic is decisive: disk is only ~6–17% of the 120B's 960 ms/token, so even a *perfect* prefetcher lifts 1.04 → at most ~1.25 tok/s. Prediction touches neither the intrinsic per-layer round-trip nor compute. **The ≥3 tok/s north-star is not reachable for streaming MoE inference on this hardware.**

## 7. Conclusion

Model residency in RAM is *not* required to run a 117B MoE on an 18 GB laptop — but *usable* speed for the flagship model is blocked by a synchronization floor intrinsic to sparse, data-dependent computation, not by any deficiency our engineering could remove. What is achievable and useful today: **30B-class MoEs at 4–7 tok/s, and 120B-class models runnable (if slow) on hardware that cannot hold them.** Our primary contribution is the *map*: five pre-registered negatives that tell the next builder exactly which streaming-MoE optimizations transfer to modern fine-grained MoEs (none of shared-base decomposition, speculative amortization, batched fetch, resident banking, or native fetch — at usable-speed granularity) and one that partially does (temporal-locality caching), plus the scale-dependence of routing predictability. We release the engine, the microbenchmarks, and every negative result so that this map costs the community nothing to reuse.

## Reproducibility

All code, the four-factor microbenchmarks, per-experiment reports (`docs/experiments/`), and raw measurement JSON (`docs/experiments/data/`) are in the repository. Every performance claim traces to a committed benchmark; negative results are committed alongside positives. Reference model set: OLMoE-1B-7B-0125, Qwen3-30B-A3B (3-bit and bf16), Qwen3-0.6B, GPT-OSS-120B-MXFP4-Q4 (all MLX community quantizations).
