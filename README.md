# rapidAI

> **A model does not need to be resident in RAM to think — only the part that is thinking right now does.**

rapidAI is an open research project attacking the assumption that a language model must fit in RAM to run. The real physical limit is **disk-to-memory bandwidth per generated token**, and that quantity is reducible by mathematics:

```
T_disk/token = (P_active × M_miss × B_bytes-per-param) / k_accept
```

Four independently-published ideas — MoE expert streaming with shared-base low-rank deltas, multi-token expert-routing prediction, entropy-coded weights decoded on-GPU, and speculative decoding as a disk-read amortizer — have never been composed into one system, and none has been built for Apple Silicon unified memory. rapidAI composes all four.

**North star:** GPT-OSS-120B (117B parameters) running at ≥ 3 tokens/s sustained on a stock 18 GB MacBook M3 Pro — a machine whose current ceiling is ~14-32B models.

**Limit demonstration:** a Kimi-K3-scale (2.8T) model executing a forward pass on the same machine, as proof that RAM size no longer defines the boundary.

## Status

Phase 0 — "Proof Week". Four pre-registered experiments with kill/live gates, before any engine code:

1. **Lookahead routing decay** — first measurements of t+2..t+k expert predictability on a modern MoE.
2. **Apple Silicon I/O reality** — mmap + madvise + Metal zero-copy streaming microbenchmarks.
3. **Expert delta spectrum** — rank/energy analysis of shared-base decomposition on Qwen3-30B-A3B.
4. **Draft acceptance rate** — speculative-decoding amortization factor measurement.

Design spec: [`docs/design/specs/2026-07-23-rapidai-streaming-inference-design.md`](docs/design/specs/2026-07-23-rapidai-streaming-inference-design.md)

All results — including negative ones — are committed to `docs/experiments/`.

## License

MIT. Everything here — code, measurements, write-ups — is a gift to whoever wants to build on it.
