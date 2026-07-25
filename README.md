# rapidAI

> **A model does not need to be resident in RAM to think — only the part that is thinking right now does.**

rapidAI is an open research project testing whether large Mixture-of-Experts (MoE) language models must fit in RAM to run on consumer Apple Silicon. It streams MoE expert weights from SSD on demand through a segmented-LRU cache, keeping only the attention/norm/embedding tier resident. Every claim is backed by a committed benchmark, and every research bet was decided by a **pre-registered live/kill gate** — negative results included.

## What we measured

**It runs — the thesis holds at the "it runs" level:**

| Model | Params (total / active) | On disk | Expert cache | Result |
|---|---|---|---|---|
| GPT-OSS-120B | 117B / 5.1B | ~63 GB | 8 GB | **1.04 tok/s** on an 18 GB M3 Pro (a 3.5× RAM model, coherent output) |
| Qwen3-30B-A3B | 30B / 3B | ~13 GB | 6 GB | **7.08 tok/s** (~3 GB total weight residency) |

Streamed output is **token-identical** to the fully-resident model (exact, not approximate).

**But usable speed for the flagship model is blocked by physics, not code.** After the cache absorbs the disk cost, the bottleneck is a per-MoE-layer CPU↔GPU round-trip *intrinsic to data-dependent fetch* — you cannot know which experts to gather until routing is computed. We proved this with a correct native C++/Metal fetch primitive that still could not beat the gate, and showed that expert prediction (the only lever that could remove the round-trip) is real but far too weak. **≥3 tok/s on 120B is not reachable with this approach on this hardware — and we can show you exactly why.**

## The real contribution: a map of what works and what doesn't

Five pre-registered **negative** results, each of which saves the next builder weeks:

- **Shared-base + low-rank expert deltas (D²-MoE-style) do not transfer** to fine-grained MoEs (delta ≈ raw; experts share no common base).
- **Speculative decoding *increases* per-token disk traffic** for expert streaming (0.48× speed, 1.72× bytes) — it needs the *union* of k tokens' expert sets. The project's guiding equation was corrected as a result.
- **Batched fetch, persistent expert bank, and a native C++/Metal port** each miss their speed gate — the bottleneck is the intrinsic per-layer round-trip, not language overhead.

Two positive findings new to the literature:

- **Multi-token routing locality does not decay:** expert overlap stays ~5.5–6× the random baseline through an 8-token horizon (first such measurement on modern decoder MoEs).
- **Routing predictability is scale-dependent:** a hidden-state predictor beats persistence by +9.66 pp at 120B (5B active) but ~0 at 1–3B active.

Full writeup: [`docs/paper/2026-07-rapidai-streaming-moe.md`](docs/paper/2026-07-rapidai-streaming-moe.md).
Per-phase reports and raw data: [`docs/experiments/`](docs/experiments/).

## Repository layout

- `tools/rapidai_tools/` — the streaming engine (SLRU cache, safetensors `pread` reader, disk-backed MoE layer)
- `tools/native/` — standalone MLX C++/Metal fetch extension (experiment; not on the default path)
- `tools/scripts/` — experiment runners and microbenchmarks
- `docs/paper/` — the preprint draft
- `docs/experiments/` — dated reports + committed measurement JSON
- `docs/design/specs/` — the design spec and thesis

## Reproducing

```bash
uv venv && uv pip install -r tools/requirements.txt
uv pip install -e tools
cd tools && ../.venv/bin/python -m pytest tests/ -q     # 32 tests
```

Benchmarks need MLX-community model weights under `models/` (gitignored); see `tools/scripts/download_models.py`.

## License

MIT. The engine, the measurements, and the negative results are a gift to whoever builds on them.
