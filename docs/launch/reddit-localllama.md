# r/LocalLLaMA draft

**Title:** Streaming MoE experts from SSD on an 18 GB M3 Pro: 30B at ~7 tok/s, 117B runs at ~1 tok/s, and an honest map of what didn't work

---

Posting this here because this sub actually reads the numbers and hates hype, which is exactly the audience I want. This is a negative-leaning result and I'm not going to pretend otherwise.

**Setup.** 18 GB M3 Pro. The idea: for a sparse MoE you only touch a handful of experts per token, so keep the attention/norm/embedding tier resident and stream the routed expert weights off the SSD on demand through a segmented-LRU cache. Reader is `pread` into a shared Metal buffer, not mmap faulting (mmap page-fault path caps at ~1.4–1.9 GB/s on this machine; pread into a buffer pool with the GPU consuming concurrently sustains ~4.16 GB/s — that inversion was the first real finding).

**What works.** Qwen3-30B-A3B (3-bit), process-isolated, O(1) cache:

| expert cache | tok/s | bytes/token | hit rate |
|---|---|---|---|
| 2 GB | 6.75 | 310 MB | 61% |
| 4 GB | 4.52 | 100 MB | 87% |
| 6 GB | 7.08 | 33 MB | 96% |

~3 GB total weight residency for a 30B. Streamed output is token-identical to the fully-resident model (verified on OLMoE greedy). GPT-OSS-120B (117B, ~63 GB on disk) runs at 1.04 tok/s, 64% hit, 680 MB/token, on 8 GB of expert cache. A model 3.5× the machine's RAM, decoding coherently.

**What doesn't — and this is the useful part.** My target was ≥3 tok/s on the 120B. Missed it, and here's why, measured with pre-registered live/kill gates:

- Once hit rate is high, disk isn't the bottleneck. It's a per-MoE-layer CPU↔GPU round-trip intrinsic to data-dependent fetch — you can't know which experts to gather until routing computes. Wrote a native C++/Metal fetch primitive (nanobind MLX extension, pread straight into unified-memory slots, single-stream Metal signal→listener→wait, zero Python sync) specifically to rule out Python overhead. Still only 1.136× vs the Python path, against a 1.5× gate. The Metal round-trip just re-labels Python's sync.
- Speculative decoding *increases* disk traffic for expert streaming: verifying k drafted tokens needs the union of their expert sets in one pass (routing overlap is only ~46%). Measured 0.48× tok/s, 1.72× bytes at 4 GB despite 59% draft acceptance. Forced a correction to my own guiding equation — k_accept only divides the resident dense/shared tier, not routed experts.
- D²-MoE-style shared-base + low-rank expert deltas don't transfer to fine-grained MoEs. On Qwen3-30B's 128 experts, 25%-rank SVD of mean-subtracted deltas captures 0.564 energy (gate 0.70) and delta ≈ raw everywhere. No shared base to factor out.
- Batched fetch and a persistent resident expert bank each miss their gates too (1.05–1.43× slower, and 1.124× vs 1.15×).

**Two positive findings.** Multi-token routing overlap doesn't decay — stays ~5.5–6× random through t+8 (first such measurement on modern decoder MoEs I'm aware of). And routing predictability is scale-dependent: a hidden-state predictor beats persistence by +9.66 pp at 120B (5B active) but ~0 at 1–3B active. Real signal, but it only covers ~23% of on-critical-path misses, and disk is only ~6–17% of the 120B's 960 ms/token — so even a perfect prefetcher tops out around 1.25 tok/s. That's the arithmetic that killed the north star.

Everything's committed, negatives included, MIT. If you run it on your own Apple Silicon, please drop a benchmark report (chip + RAM, model, cache size, tok/s, hit rate) — I've got exactly one hardware data point and I want to see where the wall actually sits.

Repo + paper: [link]
</content>
