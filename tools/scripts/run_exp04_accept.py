"""Experiment 0.4 — speculative-decoding draft acceptance length.

Measures the mean accepted-run length k when a small RAM-resident draft model
proposes tokens and the MoE target verifies them. k is the I/O amortization
factor in the four-factor equation: target weights are read once per accepted
run instead of once per token.

Usage:
  .venv/bin/python tools/scripts/run_exp04_accept.py --target models/qwen3-30b-a3b-4bit --draft models/qwen3-0.6b-4bit --tag qwen3_30b
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mlx_lm import load, stream_generate

ROOT = Path("/Volumes/x9/rapidAI")
NUM_DRAFT = 4

PROMPTS = {
    "prose": "Write a detailed essay about the industrial revolution.",
    "code": "Implement a thread-safe LRU cache in Python with unit tests.",
    "math": "Prove that the sum of the first n odd numbers equals n squared.",
    "chat": "Give me a 7-day Istanbul travel itinerary with restaurant suggestions.",
}


def accepted_runs(flags: list) -> list:
    """Lengths of maximal runs of consecutive draft-accepted tokens."""
    runs, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    model, tok = load(str(ROOT / args.target))
    draft, _ = load(str(ROOT / args.draft))

    per_domain_k, per_domain_frac = {}, {}
    for name, p in PROMPTS.items():
        messages = [{"role": "user", "content": p}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        flags = [
            r.from_draft
            for r in stream_generate(
                model, tok, prompt=prompt, max_tokens=args.max_tokens,
                draft_model=draft, num_draft_tokens=NUM_DRAFT)
        ]
        runs = accepted_runs(flags)
        per_domain_k[name] = float(np.mean(runs)) if runs else 0.0
        per_domain_frac[name] = float(np.mean(flags)) if flags else 0.0
        print(f"{name}: mean_run={per_domain_k[name]:.2f} draft_frac={per_domain_frac[name]:.2f}")

    overall = float(np.mean(list(per_domain_k.values())))
    out = {
        "experiment": "0.4 draft acceptance rate",
        "target": args.target,
        "draft": args.draft,
        "num_draft_tokens": NUM_DRAFT,
        "mean_accepted_run_by_domain": per_domain_k,
        "draft_token_fraction_by_domain": per_domain_frac,
        "overall_k": overall,
        "gate": "live" if overall >= 2.5 else ("kill" if overall < 1.5 else "gray"),
    }
    (ROOT / f"docs/experiments/data/exp04_accept_{args.tag}.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
