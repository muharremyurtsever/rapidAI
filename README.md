# rapidAI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21551120.svg)](https://doi.org/10.5281/zenodo.21551120)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A model doesn't need to live in RAM to think — only the part that's thinking right now does.**

I've got a MacBook M3 Pro with 18 GB of RAM, and for months I kept hitting the same wall. The models I actually wanted to run wouldn't fit. Not "run slowly" — just wouldn't load at all. So I started poking at a stubborn question: does a language model really have to be resident in RAM to run, or is that just how everyone happens to do it?

rapidAI is where that question went. It's an open research project that streams Mixture-of-Experts (MoE) expert weights off the SSD on demand, keeping only the attention/norm/embedding tier permanently in memory. The bet: for a sparse MoE, you only touch a handful of experts per token anyway, so why hold all of them hostage in RAM?

Here's the honest headline up front, because this is a negative-leaning result and its whole value is that it's true: **the thesis works — you really can run models far bigger than your RAM — but the flagship goal (a 117B model at chat speed) turned out to be blocked by physics, not by sloppy code.** I'll show you exactly where the wall is.

Every number below traces to a committed benchmark. Every research bet was decided by a **pre-registered live/kill gate** — the threshold written down *before* the experiment ran. Negatives are committed right alongside the wins. That discipline is the point.

## What actually changed — in plain language

Forget the jargon for a second. Here's the before and after on my 18 GB laptop, standard tooling vs. rapidAI:

| | Before (llama.cpp / MLX, no streaming) | After (rapidAI streaming) |
|---|---|---|
| **Qwen3-30B-A3B** (a 30B MoE) | Right at the edge. The 4-bit build (~17 GB) runs the machine out of memory and won't start; the 3-bit (~13 GB) loads but leaves almost nothing for anything else. | Runs **comfortably** on 2–6 GB of expert cache (~3–7 GB total weight residency), at roughly **4.5–7 tokens/second** — a handful of words a second, fine for a chat. |
| **GPT-OSS-120B** (a 117B MoE, ~63 GB on disk) | Flat-out impossible. You'd need a 64 GB+ machine. It won't even open. | **Runs.** A model nearly 4× bigger than the laptop's entire RAM produces coherent text at about **1 token/second** — roughly a word a second. Slow, patient, "leave-it-running" speed. But it works where before it wouldn't even start. |

So: can you now run bigger models on a small Mac? Yes — with a caveat I want to be dead straight about. Mid-size MoEs (the 30B class) become genuinely comfortable. Very large ones (100–120B) become *possible but slow* — you'll wait, and you'll feel it. 120B at ~1 tok/s is not a chat speed. It's a "kick it off and go make coffee" speed. I'm not going to dress that up, because the honesty is the whole selling point.

Speed intuition: ~7 tok/s is about five words a second; ~1 tok/s is about one word a second.

One more thing worth saying loudly: the streamed output is **token-identical** to the fully-resident model. Same weights, same math — streaming only changes *where the weights live while they wait*, not what the model computes. This isn't an approximation or a lossy trick.

## What we measured

| Model | Params (total / active) | On disk | Expert cache | Result |
|---|---|---|---|---|
| GPT-OSS-120B | 117B / 5.1B | ~63 GB | 8 GB | **1.04 tok/s** on an 18 GB M3 Pro — a 3.5×-RAM model, coherent output |
| Qwen3-30B-A3B | 30B / 3B | ~13 GB | 6 GB | **7.08 tok/s** (~3 GB total weight residency), 96% cache hit |

The 30B sweep, process-isolated, O(1) cache (`docs/experiments/phase1a-report.md`, `phase2-report.md`):

| expert cache | tok/s | bytes/token | hit rate |
|---|---|---|---|
| 2 GB | 6.75 | 310 MB | 61% |
| 4 GB | 4.52 | 100 MB | 87% |
| 6 GB | **7.08** | 33 MB | 96% |

## Where the wall is — and why it's physics

For the 30B, streaming is a clear win. For the 120B, it runs but stays stuck around 1 tok/s, and I spent real effort finding out why before I'd let myself write "unreachable."

Once the cache is big enough that disk stops being the bottleneck, the thing that dominates is a per-MoE-layer CPU↔GPU round-trip that's *intrinsic to data-dependent fetch*. You literally cannot know which experts to gather until the router has run — so every MoE layer forces a synchronization. I proved this wasn't a language-overhead problem by writing a correct native C++/Metal fetch primitive (no Python in the hot path at all). It still couldn't beat the gate. The round-trip just moved from Python's sync to a Metal listener's; the latency was the same order.

The only lever that could remove that dependency is prediction — guess a future token's experts from the current hidden state and prefetch them early. I measured whether that signal actually exists. It does, and it's a genuinely interesting finding (see below), but it's far too weak to close the gap. The arithmetic is brutal: disk is only ~6–17% of the 120B's ~960 ms/token, so even a *perfect* prefetcher lifts 1.04 to at most ~1.25 tok/s. Prediction attacks the smallest term in the budget.

**So ≥3 tok/s on a 120B model, on this hardware, with this approach, is not reachable.** And I can show you the receipts.

## The real contribution: a map of what works and what doesn't

The demo is fun, but the durable value here is a set of pre-registered **negative** results — each one saves the next person weeks of chasing a dead end.

Five negatives:

- **Shared-base + low-rank expert deltas (D²-MoE-style) don't transfer** to fine-grained MoEs. On Qwen3-30B's 128 small experts, the delta spectrum is essentially the raw spectrum — the experts share no common base to factor out. That trick is a Mixtral-class thing.
- **Speculative decoding *increases* per-token disk traffic** for expert streaming (0.48× speed, 1.72× bytes at 4 GB) — verifying k drafted tokens needs the *union* of their expert sets in one pass. This one actually forced a correction to the project's guiding equation.
- **Batched fetch, a persistent resident expert bank, and a native C++/Metal port** each miss their speed gate. The bottleneck is the intrinsic per-layer round-trip, not the language it's written in.

Two positive findings I haven't seen in the literature:

- **Multi-token routing locality doesn't decay.** Expert overlap between token *t* and token *t+k* stays ~5.5–6× the random baseline all the way out to an 8-token horizon — the first such measurement on modern decoder MoEs.
- **Routing predictability is scale-dependent.** A hidden-state predictor beats simple persistence by +9.66 percentage points at 120B (5B active) but by roughly nothing at 1–3B active. The signal *emerges with scale* — real, but still under the useful threshold.

Full writeup: [`docs/paper/2026-07-rapidai-streaming-moe.md`](docs/paper/2026-07-rapidai-streaming-moe.md).
Per-phase reports and raw measurement JSON: [`docs/experiments/`](docs/experiments/).

## Help me find where the wall actually sits

This is the warm ask. I ran everything on one machine — an 18 GB M3 Pro. I genuinely don't know how the numbers move on an M1 Max with 64 GB, or an M4, or a base M2 with 8 GB. The whole thesis is about the relationship between RAM, model size, and speed, and I've got exactly one data point on that curve.

If you've got any Apple Silicon Mac: clone this, run the benchmark, and tell me what you see. Model, expert-cache size, tok/s, hit rate, your chip and RAM. Open a [benchmark report issue](../../issues/new?template=benchmark-report.md) — there's a template ready. I want to map where this wall sits across real hardware, not just mine.

## Repository layout

- `tools/rapidai_tools/` — the streaming engine (segmented-LRU cache, safetensors `pread` reader, disk-backed MoE layer)
- `tools/native/` — standalone MLX C++/Metal fetch extension (experiment; not on the default path)
- `tools/scripts/` — experiment runners and microbenchmarks
- `docs/paper/` — the preprint draft
- `docs/experiments/` — dated reports + committed measurement JSON
- `docs/launch/` — announcement drafts and arXiv submission notes
- `docs/design/specs/` — the design spec and full thesis

## Reproducing

```bash
uv venv && uv pip install -r tools/requirements.txt
uv pip install -e tools
cd tools && ../.venv/bin/python -m pytest tests/ -q     # 32 tests
```

Benchmarks need MLX-community model weights under `models/` (gitignored); see `tools/scripts/download_models.py`.

## Author

Muharrem Yurtsever — full-stack developer. This is a personal research project.
LinkedIn: [muharremyurtsever](https://www.linkedin.com/in/muharremyurtsever/) · ORCID: [0009-0007-1234-7844](https://orcid.org/0009-0007-1234-7844)

## Citation

Preprint (Zenodo, CC BY 4.0): [https://doi.org/10.5281/zenodo.21551120](https://doi.org/10.5281/zenodo.21551120)

```bibtex
@misc{yurtsever2026rapidai,
  title        = {Streaming Mixture-of-Experts Inference on Consumer Apple Silicon:
                  A Measured Map of What Works and What Doesn't},
  author       = {Yurtsever, Muharrem},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21551120},
  url          = {https://doi.org/10.5281/zenodo.21551120}
}
```

## License

MIT. The engine, the measurements, and especially the negative results are a gift to whoever builds on them.

