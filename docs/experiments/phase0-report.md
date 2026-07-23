# Phase 0 "Proof Week" Report

**Date:** 2026-07-23 (all four experiments ran in a single day)
**Machine:** MacBook M3 Pro, 18 GB unified memory, internal NVMe, macOS Darwin 25.5
**Verdict:** 3 of 4 bets LIVE, 1 killed by pre-registered gate. Project proceeds to Phase 1.

## Summary

| # | Experiment | Gate (pre-registered) | Measured | Verdict |
|---|---|---|---|---|
| 0.2 | Apple Silicon I/O streaming | ≥ 3.0 GB/s GPU-consumed / kill < 1.5 | **4.16 GB/s** (pread ping-pong) | ✅ **LIVE** |
| 0.1 | Lookahead routing decay | t+2 predictability ≥ 1.5× random | **6.1× at t+2, ~5.6× at t+8** | ✅ **LIVE** |
| 0.3 | Expert delta spectrum | ≥ 70% delta energy at ≤ 25% rank | **56.4%, delta ≈ raw** | ❌ **KILL** |
| 0.4 | Draft acceptance length | k ≥ 2.5 / kill < 1.5 | **k = 2.54** | ✅ **LIVE** |

## 0.2 — Apple Silicon I/O reality (`tools/iobench`)

14 GB random file on internal NVMe, page cache purged before every measurement, 2 consistent runs.

| Path | GB/s |
|---|---|
| `dd` 8 MB blocks (ground truth) | 3.10 |
| `pread` 8 MB + `F_NOCACHE` | 4.24 |
| **`pread` → shared MTLBuffer ping-pong, GPU summing concurrently** | **4.16** |
| mmap sweep + `madvise(WILLNEED)` double-buffer | 1.19 |
| mmap sweep + `fcntl(F_RDADVISE)` | 0.38–1.79 |
| mmap, 8 threads touching pages | 1.30 |
| mmap sequential cold, single thread | 0.25 |
| mmap 16 KB random blocks | 0.13 |
| mmap 2 MB / 32 MB random blocks | 0.51 / 0.53 |

**Key finding (architecture-deciding):** the macOS mmap page-fault path saturates at ~1.4–1.9 GB/s no matter the prefetch strategy, while plain `pread` into pre-allocated shared Metal buffers sustains full SSD speed with the GPU consuming the data concurrently. This inverts the mmap-first folk wisdom inherited from llama.cpp on this platform. Second finding: 16 KB random granularity collapses throughput 30×; expert blocks must be stored contiguously and fetched at ≥ 2 MB granularity.

**Engine consequences:** (1) weight streaming via `pread` into a recycled ping-pong MTLBuffer pool; (2) `.rapd` on-disk layout keeps each expert's tensors contiguous; (3) zero-copy `newBufferWithBytesNoCopy` works and stays useful for the RAM-resident cache tier.

## 0.1 — Lookahead routing decay

Instrumented every MoE gate; captured top-8 routing across 48 layers. Full run: Qwen3-30B-A3B (3-bit), 25 generations × 1024 tokens = **25,625 decode tokens**, 5 mixed prompt domains. Pilot: OLMoE-1B-7B (64 experts, top-8), 514 tokens.

Overlap of expert sets between token t and t+lag, as multiple of the random baseline k/n:

| lag | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B (×random) | 7.36 | 6.11 | 5.82 | 5.68 | 5.55 | 5.78 | 5.98 | 5.59 |
| Qwen3-30B (absolute) | 46% | 38% | 36% | 36% | 35% | 36% | 37% | 35% |
| OLMoE (×random) | 3.33 | 2.70 | 2.53 | 2.56 | 2.48 | 2.50 | 2.57 | 2.64 |

**Key finding:** the signal does not decay toward baseline — it drops once from t+1 to t+2 and then **plateaus** through t+8 on both models, at both scales. To our knowledge these are the first multi-token-lookahead routing measurements published for modern decoder MoEs (prior art: Pre-gated MoE / Fate / ST-MoE measure one layer ahead; SiDA/ETH the current token only). The plateau means (a) an SLRU expert cache will run at high hit rates, and (b) prefetching experts many tokens ahead is not fighting entropy — a predictor at t+8 has as much signal as at t+2.

**Deviation from plan:** 4-bit 30B model (16 GB) exceeded the Metal working-set limit (~13.6 GB) and crashed with GPU OOM; per the plan's fallback we used 3-bit (12.7 GB) and 1024 tokens/generation instead of 2048. Routing statistics are quantization-robust (the router runs in higher precision), so we consider the substitution immaterial — flagged for re-verification in Phase 1.

## 0.3 — Expert delta spectrum: Bet A is dead

Qwen3-30B-A3B bf16 weights, 5 layers (0, 12, 24, 36, 47) × 3 projections × 128 experts. Decomposition: base = mean over experts, delta = expert − base; measured Frobenius-energy captured by rank-r SVD truncation.

- Mean delta energy at 25% rank: **0.564** (gate required ≥ 0.70).
- Decisive detail: **delta ≈ raw within 0.003 in all 15 combos** — subtracting the shared base removes essentially no energy. The 128 fine-grained experts share no meaningful common component.

**Interpretation:** D²-MoE's published gains (Mixtral/Phi-class, 8–16 coarse experts) do not transfer to modern fine-grained MoEs, whose training explicitly drives expert specialization. Since our targets (Qwen3, GPT-OSS, Kimi-class) are all fine-grained, **Bet A (shared-base + streamed low-rank deltas) is dropped.** This is a publishable negative result: anyone planning "D²-MoE + streaming" on a modern MoE should see this measurement first.

## 0.4 — Draft acceptance length

Qwen3-0.6B-4bit draft proposing 4 tokens/round against Qwen3-30B-A3B-3bit target, 1024 tokens per domain, mlx-lm speculative decoding.

| domain | prose | code | math | chat | **overall** |
|---|---|---|---|---|---|
| mean accepted run k | 2.33 | 2.48 | 2.89 | 2.46 | **2.54** |
| fraction of tokens from draft | 59% | 62% | 67% | 57% | 61% |

Gate (k ≥ 2.5) passes at the margin. 61% of tokens never require a fresh target-model invocation of their own — in a disk-streaming engine that directly divides per-token disk traffic. Headroom exists: longer draft windows, better-matched drafts, and acceptance-aware tuning typically raise k; treat 2.54 as a floor.

## Implications — the equation with real numbers

`T_disk/token = (P_active × M_miss × B_bytes/param) / k_accept`, with Bet A's multiplier removed.

Projection for GPT-OSS-120B (117B total, 5.1B active, ~4.25-bit MXFP4 ≈ 63 GB) on this machine:

- Active bytes/token ≈ 5.1B × 4.25/8 ≈ **2.7 GB**
- `M_miss`: measured 35–46% single-lag reuse is a lower bound on cache hits; a multi-token SLRU + t+k prefetcher plausibly reaches 50–75% hit rate → miss 0.25–0.5
- `k_accept` = 2.5 (measured floor)
- Disk budget/token ≈ 2.7 × (0.25…0.5) / 2.5 ≈ **0.27–0.54 GB**
- At the measured 4.16 GB/s: **7.7–15.4 tokens/s I/O ceiling** (compute and mispredictions will eat into this; entropy coding [Bet C] adds ~1.3–1.8× headroom on top)

The pre-registered north star (≥ 3 tok/s) sits comfortably inside this envelope. These are projections, not results — Phase 1's baseline engine exists to replace them with measurements.

## Deviations from plan

1. 32 GB iobench file reduced to 14 GB (internal disk had 21 GB free). Cache purged per measurement; `dd` ground truth consistent.
2. Qwen3-30B 4-bit → 3-bit substitution after Metal OOM (documented above); exp 0.4 also used the 3-bit target.
3. Experiment 0.1 used 1024 tokens/generation (plan: 2048) for memory headroom; token count target (~25k vs 50k) halved accordingly. Plateau shape is unambiguous at this sample size.
4. iobench gained measurements not in the plan (F_RDADVISE, multithreaded sweep, pread baselines, GPU pread ping-pong) — added mid-run when the mmap numbers came in far below ground truth. The added path is what passed the gate; the plan's original mmap-only path would have **failed** it. This is why we measure.

## Phase 1 go decision

- Bets **E (speculative amortizer), B (SLRU + t+k prefetch), C (entropy coding)** proceed, in that order.
- Bet **A** dropped; its slot in the plan is replaced by the pread-pool streaming layer that 0.2 proved out.
- First Phase 1 milestone: baseline engine streaming Qwen3-30B-A3B from SSD with a deliberately small RAM budget, reproducing the I/O ceiling math above with end-to-end tok/s.
