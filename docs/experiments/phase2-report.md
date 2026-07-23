# Phase 2 Report — Speculative Amortizer (Bet E): KILLED, with a corrected equation

**Date:** 2026-07-24
**Machine:** MacBook M3 Pro 18 GB (reference)
**Pre-registered gate: FAILED → Bet E dropped for MoE expert streaming.** A measurement-methodology bug was also found and fixed; corrected Phase 1a numbers are *better* than first reported.

## Methodology correction (affects Phase 1a numbers)

The original sweep ran all budgets in one Python process; cross-iteration state pollution depressed tok/s (and zeroed draft-acceptance flags in two runs). All headline numbers below are **process-isolated reruns** (one budget per process). `bytes_read_per_token` and hit rates were unaffected; only tok/s was under-measured.

**Corrected Phase 1a table (Qwen3-30B-A3B 3-bit, streamed, 256 decode tokens):**

| expert-cache budget | tok/s (corrected) | bytes/token | hit rate |
|---|---|---|---|
| 2 GB | 2.91 | 310 MB | 60.7% |
| 4 GB | 2.53 | 100 MB | 87.3% |
| 6 GB | **4.06** | 33 MB | 95.8% |

New headline: **a 30B MoE decodes at 4.06 tok/s on an 18 GB Mac with 6 GB of expert cache.** (Curiosity: 2 GB outruns 4 GB because a smaller wired cache leaves more RAM to the OS page cache holding the 13 GB weight file — whole-machine memory economics matter and will be modeled in Phase 1b.)

## Bet E result (speculative decoding over the streamed target)

Qwen3-0.6B draft (resident) proposing 4 tokens/round; streamed 30B verifies. Acceptance was healthy and constant (59.4% of tokens draft-originated) — speculation *worked*; it just doesn't pay here:

| budget | spec tok/s | non-spec tok/s | spec bytes/token | non-spec bytes/token |
|---|---|---|---|---|
| 2 GB | 1.13 | 2.91 | 690 MB | 310 MB |
| 4 GB | 1.21 | 2.53 | 172 MB | 100 MB |
| 6 GB | 2.67 | 4.06 | 57 MB | 33 MB |

Gate required ≥1.5× tok/s and ≤0.6× bytes at 4 GB; measured **0.48× tok/s and 1.72× bytes. KILL.**

## Why — the equation was wrong about MoE

The four-factor equation assumed `k_accept` divides disk traffic: verify k tokens in one pass → read weights once per k tokens. **That holds for dense/shared weights only.** In a fine-grained MoE, each token routes to a mostly-distinct expert set (46% single-lag overlap), so a 5-token verification batch needs the **union** of five expert sets — ~3-4× the experts of a single token — in one call, under one cache. Sequential decoding fetches nearly the same total bytes but reuses the cache *between* tokens; batched verification pays the union up front and evicts harder. Net effect: more bytes, lower hit rate, plus draft compute on top.

**Corrected equation:**

```
T_disk/token = P_active(routed) × M_miss × B_bytes/param        [experts]
             + P_shared × M_miss_shared × B / k_accept          [dense/shared tier — small]
```

`k_accept` survives only on the small shared tier (attention/norms/embeddings — which we keep resident anyway). For expert streaming, the surviving levers are **M_miss (Bet B: SLRU + prefetch)** and **B_bytes/param (Bet C: entropy coding)**, on top of MoE sparsity itself. Two of our four bets (A, E) are now measured dead — this is the report card of pre-registered gates doing their job, and both negatives are publishable insights (nobody has published "speculative decoding hurts disk-streamed fine-grained MoE" with numbers, to our knowledge).

## Roadmap after two kills

1. **Phase 1b — C++/Metal port of the hot path** (fetch → compact bank → gather): Python orchestration still costs ~200-350 ms/token at high hit rates; the I/O floor at 6 GB is ~8 ms/token. Ceiling if ported well: >10 tok/s for the 30B on this machine.
2. **Bet B completion** — t+k predictor prefetching into the SLRU during attention compute (hides the residual miss latency; routing plateau says signal is there).
3. **Bet C** — entropy-coded expert blocks, Metal-decoded (multiplies effective SSD bandwidth 1.3-1.8×).
4. Then GPT-OSS-120B (Phase 6 north star) with the corrected equation.
