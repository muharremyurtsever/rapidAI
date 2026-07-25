# Novelty Deep-Research Report — 2026-07-23

Method: structured literature review across 5 search angles → 22 primary sources → 109 candidate claims → 25 verified against their primary sources. 24 confirmed, 1 refuted.

Question: are our four bets for reducing per-token disk I/O in MoE inference on consumer Apple Silicon already done?

## Verdicts

### Bet A — Shared base + low-rank expert deltas, streamed: PARTIALLY DONE

- **D²-MoE** (ICML 2025, [arXiv:2502.17298](https://arxiv.org/abs/2502.17298), [code](https://github.com/lliai/D2MoE)) implements the exact decomposition: Fisher-merged shared base + SVD low-rank per-expert deltas, 40-60% training-free compression on Mixtral/Phi-3.5/DeepSeek/Qwen2. **Validates the math** (inter-expert similarity, low-rank deltas). Verified 3-0.
- BUT: static in-memory compression only; all experiments on A100s; SSD/offload/prefetch never appear in the method (verified 2-1).
- **Low-Rank Compensation MoE** (Dec 2025, [arXiv:2512.17073](https://arxiv.org/abs/2512.17073)) streams low-rank compensators for routed experts — but per-expert quantization residuals, **no shared cross-expert base**, no Apple Silicon. Verified 3-0.
- **Our claimable delta:** base resident in RAM + deltas streamed from SSD, per-token I/O accounting, consumer Apple Silicon.

### Bet B — t+2..t+k expert prediction for prefetch: PARTIALLY DONE (crowded)

- One-layer-ahead, same token: **Pre-gated MoE** (ISCA 2024, requires fine-tuning), **Fate** (arXiv:2502.12224; 78.79% base decode accuracy, 97.15% with over-prefetch; 4.1-4.5x speedups on RTX 3090/1080Ti), **ST-MoE** (arXiv:2606.15453; 85%, history + cross-layer tables). All verified 3-0 (one 2-1).
- Current-token: **SiDA-MoE** (MLSys 2024, ~99% top-3 on Switch), **ETH pre-attention predictor** (arXiv:2511.10676-adjacent, 93-98% on DeepSeek-V2-Lite/Qwen3-30B/Phi-mini-MoE, 2 linear layers). Verified 3-0.
- Multi-token exists ONLY via speculative decoding structure: **SP-MoE** (arXiv:2510.10302, draft-target correspondence, 1.07-3.5x). Verified 3-0.
- **Nobody predicts experts for future tokens from hidden states, draft-free.** Nobody measures how the ~2x-over-random consecutive-token overlap decays at t+k. **Both open.**

### Bet C — Sub-2-bit entropy-coded weights, Metal in-shader decode: MOSTLY OPEN

- **DFloat11** ([arXiv:2504.11651](https://arxiv.org/pdf/2504.11651)): Huffman-coded BF16 exponents, on-GPU decode, lossless ~11 bits/param, CUDA. Verified 3-0.
- **EntQuant** (arXiv:2601.22787): parallelized ANS decode in the GPU forward pass — CUDA, not sub-2-bit.
- **ZipServ** (ASPLOS 2026): fused decompress+GEMM into tensor-core registers — lossless BF16, CUDA.
- **QVAC Fabric BitNet** (github.com/tetherto/qvac-rnd-fabric-llm-bitnet): first Metal/Vulkan ternary backend — fixed-width TQ1_0/TQ2_0 packing (1.0-2.0 bits), **no entropy coding**.
- **Our claimable delta:** entropy-coded sub-2-bit blocks + Metal threadgroup decode + SSD-bandwidth-amplification framing. Caveat: this angle had the thinnest verification coverage (NestQuant/ZipNN not adversarially checked) — re-scan before any paper claim.

### Bet E — Speculative decoding as disk-read amortizer: OPEN AS FRAMING

- SP-MoE uses speculation for prefetch (latency hiding), not for amortizing SSD reads across k accepted tokens with target-on-disk. The "draft lives in RAM, target sleeps on SSD" architecture is unclaimed on Apple Silicon.

## Measured facts to reuse

- Consecutive-token expert overlap ≈ 2× the random K²/N baseline (DeepSeek/Qwen/Mixtral; ST-MoE). Verified 3-0.
- Aggregate expert usage near-uniform (normalized entropy 0.976, DeepSeek-V2-Lite) → predictors must use sequence-conditional signals, not popularity. Verified 2-1 (small sample).
- Disk→GPU expert load 5-8.4× slower than RAM→GPU (99 MB / 6 experts: 33.5-49.8 ms vs 4.0-9.5 ms) — NVIDIA/PCIe numbers; ~40 ms/token I/O caps decode at 20-25 tok/s. Verified 3-0. **Must re-measure on Apple Silicon (Phase 0.2).**

## Refuted during verification

- "Prediction-driven prefetching + expert replication yields 3x speedup at 90-95% quality" (arXiv:2605.11537) — refuted 0-3. Treat that paper's headline numbers skeptically.

## Caveats

1. Most sources are preprints; only D²-MoE, Pre-gated MoE, SiDA-MoE peer-reviewed. Self-reported numbers.
2. All hardware numbers are NVIDIA/PCIe — Apple Silicon UMA ratios unknown (that gap is our opportunity and our risk).
3. The A-space is closing fast (two adjacent papers in 10 months). Re-run a fresh literature scan before any preprint submission.
4. Fate's 97.15% is the over-prefetch variant; use 78.79% when estimating predictor difficulty.

## Open questions carried into Phase 0

1. Does the 2× overlap persist at t+2..t+8, and how fast does predictability decay? → Experiment 0.1
2. Real SSD→UMA→Metal expert-load latency/bandwidth on M3 Pro? → Experiment 0.2
3. Are A and C multiplicative or overlapping (entropy-coding already-low-rank deltas)? → Phase 5 ablation
4. Does acceptance length k survive a quantized+delta-approximated target? → Phase 4 exit criteria
