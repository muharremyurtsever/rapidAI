# arXiv submission notes (for Muharrem)

Practical notes for putting the paper on arXiv yourself. I can't submit it for you — arXiv needs your own account and identity — so this is the checklist plus everything I could prepare ahead of time.

## The honest state of it

- The paper lives at `docs/paper/2026-07-rapidai-streaming-moe.md` as **Markdown**. arXiv does **not** accept Markdown. You'll need a PDF, and for a clean listing, ideally the LaTeX source (arXiv prefers `.tex` and will compile it their side; PDF-only is allowed but discouraged and blocks their processing niceties).
- **This is the one remaining manual step.** Converting the markdown to LaTeX/PDF is quick but I didn't scaffold a full `.tex` in this pass — the tables and the corrected equation need a careful eye so nothing drifts from the committed numbers. Two easy paths:
  - **Pandoc:** `pandoc docs/paper/2026-07-rapidai-streaming-moe.md -o paper.pdf` (needs a TeX distribution like MacTeX). Gives you a PDF immediately; add `--to=latex -o paper.tex` to get editable source to clean up before submitting.
  - **Overleaf:** paste the content into a blank `article` template, fix the tables by hand, compile. Slower but you see exactly what arXiv will show.
- Whichever route: after converting, **diff every number against the committed reports** (`docs/experiments/*`). The prose was humanized; the tables must stay byte-exact.

## Account + endorsement

- You need an arXiv account tied to your real name/affiliation.
- As a **first-time submitter** to these categories you'll almost certainly need an **endorsement** — arXiv requires an established author in the category to vouch for a new submitter. Plan for this; it can take a few days. If you know anyone who's published in cs.LG or cs.DC, ask early.
- Endorsement is per-category, so pick your primary category before requesting.

## Suggested categories

- **Primary: cs.LG** (Machine Learning) — natural home; the routing-predictability findings fit here.
- **Cross-list: cs.DC** (Distributed, Parallel, and Cluster Computing) — the streaming/systems angle.
- **Cross-list: cs.PF** (Performance) — the whole paper is a performance measurement study; strong fit for the microbenchmarks and the sync-floor finding.

A cs.LG primary with cs.DC + cs.PF cross-lists is a reasonable spread. If endorsement in cs.LG is hard to get, cs.PF or cs.DC as primary is defensible given the paper is more systems-measurement than new-model.

## Metadata to have ready

- **Title:** Streaming Mixture-of-Experts Inference on Consumer Apple Silicon: A Measured Map of What Works and What Doesn't
- **Author:** Muharrem Yurtsever
- **Comments field (optional but nice):** note it's a negative-leaning systems study with all code + data open, and link the repo.
- **License:** the repo is MIT; on arXiv pick a license you're comfortable with (CC BY 4.0 is common and lets people reuse the text; arXiv's default non-exclusive license is the minimum).

## Abstract (ready to paste — matches the current paper)

> Does a large Mixture-of-Experts (MoE) language model have to fit in RAM to run at usable speed on consumer hardware? We built a streaming inference engine for Apple Silicon to find out — it keeps a model's attention/norm/embedding tier resident and pulls MoE expert weights off the SSD on demand through a segmented-LRU cache — and we held every design decision to a live/kill gate fixed before the experiment ran. The thesis survives at the "it runs" level and then hits a wall we can locate precisely. Three positive results: (1) GPT-OSS-120B (117B parameters, 5.1B active, ~63 GB on disk) decodes coherently at 1.04 tok/s on an 18 GB machine — a model 3.5x larger than total RAM, running where it otherwise would not load at all; (2) Qwen3-30B-A3B decodes at up to 7.08 tok/s with a 6 GB expert cache (~3 GB total weight residency), comfortably usable; (3) we contribute the first multi-token expert-routing predictability measurements on modern decoder MoEs, and the routing overlap does not decay across an 8-token horizon. The other half of the paper is the more useful half: five pre-registered negative results that pin down the design space. Shared-base + low-rank expert decomposition (D2-MoE-style) does not transfer to fine-grained MoEs; speculative decoding increases per-token disk traffic for expert streaming; and three successive engine optimizations — batched fetch, a persistent expert bank, a native C++/Metal fetch primitive — each miss their speed gate for the same reason: the bottleneck is a per-layer CPU-GPU round-trip intrinsic to data-dependent fetch, not something the engineering can remove. A lightweight expert predictor can't dodge it either — the signal is real and grows with scale (+9.66 points over persistence at 120B, versus roughly zero at 1-3B active), but far too weak (23% miss coverage) to matter, and disk was never the dominant cost anyway. So the >=3 tok/s target is unreachable for streaming MoE inference on this hardware class. We release the engine, every measurement, and every negative so the next person doesn't have to re-derive this map.

(arXiv abstracts are plain text — the Unicode arrows/×/≥ and the subscripts in the equation won't render in the abstract box, so I've de-Unicoded this copy. The PDF keeps the proper symbols.)

## Order of operations

1. Convert markdown → LaTeX/PDF, verify every number against `docs/experiments/`.
2. Create/confirm arXiv account; request endorsement for your primary category if prompted.
3. Upload source, set categories (cs.LG primary; cs.DC, cs.PF cross-list), paste metadata + abstract.
4. Preview the compiled PDF on arXiv, check tables one more time.
5. Submit. There's a moderation hold (usually a day or so) before it goes live.

Nothing here is public until you do step 5 — the repo stays private and no announcement goes out until you decide.
