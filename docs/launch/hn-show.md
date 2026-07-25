# Show HN draft

**Title:** Show HN: I ran a 117B model on an 18 GB MacBook (it works, it's slow, here's the map)

---

I've got an M3 Pro with 18 GB of RAM, and I kept hitting a wall: the models I wanted to run wouldn't fit. So I went down a rabbit hole on a simple question — does a model actually have to live in RAM to run, or is that just the convention?

rapidAI streams a Mixture-of-Experts model's expert weights off the SSD on demand and keeps only the always-on parts (attention, norms, embeddings) resident. For a sparse MoE you only touch a few experts per token anyway, so most of the model can sit on disk until it's needed.

The surprising part first: GPT-OSS-120B (117B params, ~63 GB on disk — nearly 4× my entire RAM) decodes coherent text on this laptop at about 1 token/second. It runs where before it wouldn't even open. And the output is token-identical to the fully-resident model, not an approximation.

Now the honest part, because that's the whole point. 1 tok/s is a "leave it running" speed, not a chat speed, and I could not get it faster. My original goal was 3 tok/s on the 120B. I missed it — and I spent real effort proving *why* rather than hand-waving. The bottleneck turns out to be a per-MoE-layer CPU↔GPU round-trip that's intrinsic to data-dependent fetch: you can't know which experts to grab until routing runs. I even wrote a native C++/Metal fetch primitive to rule out language overhead. Same wall. It's physics on this hardware, not sloppy code.

The real deliverable is the map: five pre-registered negative results (speculative decoding actually *hurts* here; expert-delta compression doesn't transfer to fine-grained MoEs; three engine ports each miss their gate) plus two new findings on routing locality. Mid-size 30B MoEs, for what it's worth, run genuinely comfortably at 4–7 tok/s.

MIT, everything committed including the failures. If you've got any Apple Silicon Mac, I'd love a benchmark report — I want to see where this wall sits on hardware that isn't mine.

Repo + paper: [link]
</content>
