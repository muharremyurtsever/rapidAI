# Phase 3 — Bet B (t+k expert prediction) feasibility

**Date:** 2026-07-25
**Question:** Is there real *sequence-conditional* signal for predicting a future
token's routed expert set from the current token's hidden state — signal beyond
"the next token uses the same experts as this one" (persistence) — enough to make
an async prefetcher worth building?

**Why it matters:** All three Phase-1b orchestration levers (batched fetch,
persistent bank, native C++/Metal port) are measured dead. The native port proved
the bottleneck is a per-MoE-layer CPU↔GPU round-trip *intrinsic to data-dependent
fetch*: you cannot know which experts to gather until routing is computed, forcing
a sync per layer. The only way to remove that sync from the critical path is to
predict a future token's experts *ahead of time* and prefetch them asynchronously.
Prediction is only useful if it beats what the SLRU cache already exploits for free
(persistence). This experiment measures the gap before any prefetcher is built.

## Pre-registered gate (written BEFORE looking at any results)

Let `recall@k` = fraction of token t+1's actually-routed top-k experts (per MoE
layer, averaged over layers and eval tokens) that appear in the predictor's top-k'
prediction, with k' = k.

- **PERSISTENCE baseline:** predict t+1's experts = t's experts (what the cache
  already gives for free).
- **TRAINED predictor:** a lightweight linear (and 2-layer MLP cross-check) map from
  token t's layer-L gate/router input hidden state → t+1's layer-L expert logits,
  trained on the first 70% of a routing trace, evaluated on the held-out last 30%.

**GATE — Bet B is worth building iff BOTH hold on GPT-OSS-120B (the model that
matters):**
1. `recall@k(trained) − recall@k(persistence) ≥ 15 percentage points`, AND
2. `recall@k(trained) ≥ 60%` absolute.

i.e. there is real sequence-conditional signal beyond "same as last token."

Also reported (not gated, but decision-informing):
- `recall@2k` (k' = 2k) for both predictors.
- recall decay at t+2, t+3, t+4 (how deep the prefetch horizon can reach).
- **The metric that actually matters:** of token t+1's *missed* experts (those NOT
  resident in a simulated SLRU cache at the deployment budget), what fraction would
  the predictor have prefetched in time (i.e. are in its top-k')? This is the
  fraction of on-critical-path disk misses the prefetcher removes.

**STEP 2 (build async prefetch) runs ONLY if the gate passes.**
**STEP 3 (honest negative + pivot recommendation) if it fails.**

Models: GPT-OSS-120B-MXFP4-Q4 (36 layers, d=2880, 128 experts, top-4) — the model
that matters (64% hit, on-critical-path misses dominate). Cross-check: Qwen3-30B-A3B
3-bit (48 layers, d=2048, 128 experts, top-8).

---

## Results

Three models captured (per-decode-step gate-input hidden state + routed top-k
experts, every MoE layer). Trained predictor = train-only-PCA(256) → ridge from
token t's layer-L hidden state to a multi-hot of token t+lag's layer-L experts,
fit on the first 70% of the trace, evaluated on the held-out last 30%. A planted-
signal unit test confirms this predictor reaches recall 0.6+ (vs 0.25 chance) when
signal exists — so a null result here is absence of signal, not a code bug.

### The metric that actually matters — miss coverage (lead with this)

`recall@k` is the wrong headline. Persistence already scores high recall@k, but
that is exactly the autocorrelation the SLRU cache **already exploits for free** —
those experts are resident, not misses. Prefetch only helps on the experts that
are genuinely NOT in cache (on-critical-path disk misses). The decision metric is:
of token t+lag's *missed* experts, what fraction would the predictor have fetched
in time?

| model | eval hit rate | persistence miss-coverage | **trained miss-coverage (t+1)** |
|---|---|---|---|
| GPT-OSS-120B-MXFP4 @8192 MB (the model that matters) | 65.8% | 0.0% | **23.3%** (35,536 misses scored) |
| Qwen3-30B-A3B 3-bit @6144 MB | 97.0% | 0.0% | **14.6%** |
| OLMoE-1B-7B 4-bit (pilot) | ~100% | — | — (no misses at budget) |

Persistence miss-coverage is **structurally 0%**: persistence predicts the
recently-used experts, which are precisely the ones already cached — it can never
name a miss. The trained predictor names 23% of the true misses on 120B (15% on
30B). An async prefetcher built on this would leave **~77% of on-critical-path disk
misses still on the critical path** — and, as the arithmetic below shows, even
perfect coverage cannot reach ≥3 tok/s because disk is only a fraction of the
960 ms/token.

### recall@k (context — high for persistence because the cache already has it)

**GPT-OSS-120B-MXFP4-Q4** — the model the gate names (36 layers, d=2880, top-4,
2406 decode tokens, 65.8% eval hit rate). Trained miss-coverage in the last column:

| horizon | persistence@k | trained@k | Δ (pp) | persistence@2k | trained@2k | trained miss-cov |
|---|---|---|---|---|---|---|
| t+1 | 0.307 | 0.404 | **+9.66** | 0.413 | 0.560 | 23.3% |
| t+2 | 0.258 | 0.301 | +4.30 | 0.371 | 0.443 | 13.7% |
| t+3 | 0.249 | 0.262 | +1.37 | 0.358 | 0.393 | 10.5% |
| t+4 | 0.237 | 0.245 | +0.80 | 0.335 | 0.367 | 8.9% |

Hybrid predictor (hidden state + current-token experts as features, t+1): 0.395.
The signal decays fast with horizon: the +9.66pp edge at t+1 collapses to +0.80pp
by t+4, so deeper prefetch buys almost nothing.

**Qwen3-30B-A3B 3-bit** — cross-check (48 layers, d=2048, top-8, 3078 decode tokens):

| horizon | persistence@k | trained@k | Δ (pp) | persistence@2k | trained@2k |
|---|---|---|---|---|---|
| t+1 | 0.443 | 0.429 | **−1.4** | 0.539 | 0.580 |
| t+2 | 0.359 | 0.358 | −0.1 | 0.471 | 0.505 |
| t+3 | 0.342 | 0.332 | −0.9 | 0.455 | 0.475 |
| t+4 | 0.334 | 0.327 | −0.7 | 0.445 | 0.464 |

Hybrid predictor (hidden state + current experts, t+1): 0.437 — still below
persistence's 0.443. Even handed the persistence answer as a feature, the
hidden state adds no recoverable sequence signal.

(Pilot: OLMoE-1B-7B 4-bit, 162 tokens, same picture — trained@k 0.399 vs
persistence 0.410, Δ −1.1pp.)

### Gate verdict

| gate condition | GPT-OSS-120B | Qwen3-30B |
|---|---|---|
| trained@k − persistence@k ≥ 15 pp | ❌ +9.66 pp | ❌ −1.4 pp |
| trained@k ≥ 60% absolute | ❌ 40.4% | ❌ 42.9% |
| **PASS** | **❌ FAIL** | **❌ FAIL** |

**Both models fail the pre-registered gate.** The 120B misses on both conditions:
the +9.66pp edge is real but below the +15pp bar, and 40.4% absolute is below 60%.

## Interpretation — routing predictability GROWS with scale, but not enough

A genuine, honest nuance that the arithmetic still overrides:

- On OLMoE (1B active) and Qwen3-30B (3B active), the trained predictor does **not**
  beat persistence at all — Δ is ~0 or negative. The hidden state carries no
  recoverable signal about *new* experts beyond the cache's autocorrelation.
- On **GPT-OSS-120B (5.1B active) — the model that matters — a real
  sequence-conditional signal appears**: trained beats persistence by +9.66pp at
  t+1 and catches 23% of on-critical-path misses (vs persistence's structural 0%).
  Larger, more specialized MoEs route in a way the per-layer hidden state partially
  predicts. This is a new, publishable finding: **routing predictability is
  scale-dependent** — it emerges with model size rather than being present at all
  scales.

This does not save the north star. The predictor recovers a planted deterministic
signal to 0.6+ recall (unit test), so the ceiling is real, not a code artifact —
the 120B signal is just genuinely weak. Exp 0.1's 5.5–7.4× overlap is mostly the
persistence/autocorrelation the SLRU already captures for free (that is why
persistence miss-coverage is 0%); the residual *new* experts each token pulls in
are where the disk misses live, and even on the 120B only ~23% of them are
predictable one token ahead, decaying to ~9% by four tokens ahead.

### Why 23% miss-coverage cannot reach ≥3 tok/s (the arithmetic)

At 1.04 tok/s the 120B spends ~960 ms/token. Of that, on-critical-path disk is
~245 MB/token of misses ÷ 4.16 GB/s ≈ **~60 ms/token (~6–17%)**; the rest is the
intrinsic per-layer CPU↔GPU round-trip (Phase 1b native-port finding) plus compute.
A prefetcher covering 23% of misses hides ~14 ms/token → best case ~1.06 tok/s. Even
a *perfect* prefetcher (100% coverage, disk fully off the critical path) removes
only ~60 ms → **~1.07→1.25 tok/s** — the round-trip and compute floor keep it far
below 3. Prediction attacks the smallest term in the budget.

## STEP 3 — decision: do NOT build the prefetcher; honest negative

Per the pre-registered flow, the gate fails, so the async prefetcher is **not
built**. This is the honest negative result the flow demanded.

## North-star assessment — is ≥3 tok/s on GPT-OSS-120B reachable this way?

**No — not with the streaming approach on this hardware.** The evidence is now
complete and mutually reinforcing:

1. All three Phase-1b orchestration levers are dead by pre-registered gate: batched
   fetch (negative), persistent bank (1.124× < 1.15×), native C++/Metal port
   (1.136× < 1.5×). The native port proved the bottleneck is a per-MoE-layer
   CPU↔GPU round-trip **intrinsic to data-dependent fetch** — it cannot be
   engineered away in any language.
2. The only lever that could remove that dependency from the critical path — Bet B
   prediction + async prefetch — is measured real-but-insufficient: on the 120B the
   trainable signal covers 23% of on-critical-path misses at t+1 (persistence covers
   0% by construction), decaying to ~9% by t+4. Disk is only ~6–17% of the
   960 ms/token budget, so even a perfect prefetcher lifts 1.04 tok/s to at most
   ~1.25 — the intrinsic per-layer round-trip and compute dominate and neither is
   touched by prediction.

GPT-OSS-120B decoding coherently at 1.04 tok/s on an 18 GB machine (a model 3.5×
its RAM) stands as a real, publishable systems result. But the ≥3 tok/s north star
required a factor the data does not support. **Recommendation: stop chasing ≥3
tok/s via streaming and pivot to publishing what exists** — the 120B-on-18GB
demonstration, the four original Phase-0 positive measurements (I/O reality,
routing-overlap plateau, delta-spectrum kill, draft-acceptance k=2.54), the FIVE
honest negatives (Bet A delta-spectrum; the three Phase-1b orchestration levers;
Bet B prefetch), and the new **scale-dependent routing-predictability** finding
(prediction signal absent at 1–3B active, emerging at 5B active — but still below
the useful threshold). The negatives are the contribution: they tell the field
precisely which streaming-MoE optimizations do and do not transfer to modern
fine-grained MoEs, and why data-dependent expert fetch has an irreducible per-layer
sync floor that prediction cannot yet clear.

Residual upside NOT pursued here (would be separate projects, not v1 streaming):
speculative decoding (measured k=2.54 amortizer, divides disk traffic directly),
entropy-coding the expert bytes (Bet C, ~1.3–1.8× on `B_bytes`), or a smaller
target model — none restore the streaming north star for 120B on 18 GB.
