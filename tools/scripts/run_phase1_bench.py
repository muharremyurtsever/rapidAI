"""Phase 1 bench: tok/s and bytes/token vs expert-cache budget.

Usage:
  .venv/bin/python tools/scripts/run_phase1_bench.py --model models/olmoe-1b-7b-4bit --tag olmoe --budgets-mb 256,512,1024,2048,3584
  .venv/bin/python tools/scripts/run_phase1_bench.py --model models/qwen3-30b-a3b-3bit --tag qwen3_30b --budgets-mb 2048,4096,6144
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

from rapidai_tools.streamed_moe import install_streaming

ROOT = Path("/Volumes/x9/rapidAI")
PROMPT = "Write a detailed technical explanation of how TCP congestion control works."
N_TOKENS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budgets-mb", required=True)
    ap.add_argument("--max-tokens", type=int, default=N_TOKENS)
    args = ap.parse_args()
    runs = []
    for mb in [int(x) for x in args.budgets_mb.split(",")]:
        model, tok = load(str(ROOT / args.model))
        stats = install_streaming(model, str(ROOT / args.model), cache_bytes=mb << 20)
        # measure decode only: exclude prefill by starting the clock at first token
        n = 0
        t0 = None
        b0 = h0 = m0 = None
        for r in stream_generate(model, tok, prompt=PROMPT, max_tokens=args.max_tokens):
            if t0 is None:
                t0 = time.perf_counter()
                b0 = stats.bytes_read
                h0, m0 = stats.cache.hits, stats.cache.misses
            n += 1
        dt = time.perf_counter() - t0
        hits = stats.cache.hits - h0
        misses = stats.cache.misses - m0
        runs.append({
            "budget_mb": mb,
            "tokens": n,
            "tok_s": round(n / dt, 3),
            "bytes_read_per_token": int((stats.bytes_read - b0) / max(n, 1)),
            "decode_cache_hit_rate": round(hits / max(hits + misses, 1), 4),
        })
        print(json.dumps(runs[-1]), flush=True)
        del model
        mx.clear_cache()
    out = {"experiment": "phase1 bench", "model": args.model, "runs": runs}
    (ROOT / f"docs/experiments/data/phase1_bench_{args.tag}.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
