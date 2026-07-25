"""Phase 1b native-port e2e A/B driver (NEGATIVE result — kept reproducible).

Runs ONE variant per process (project benchmark rule) so the pre-registered
interleaved protocol is: fresh process per run, alternate variants, >= 3
trials each, compare medians. Example (as run for the report):

  for i in 0 1 2; do
    for v in percall native; do
      .venv/bin/python tools/scripts/run_phase1b_native_bench.py \
          --bank $v --tag qwen3_30b_nativeAB_${v}_r$i \
          --model models/qwen3-30b-a3b-3bit --budget-mb 6144
    done
  done

Verdict (2026-07-25): native 5.802 vs percall 5.109 tok/s medians = 1.136x,
below the pre-registered 1.5x gate -> library keeps the Python path; the
native extension stays as a standalone experiment.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

ROOT = Path("/Volumes/x9/rapidAI")
PROMPT = "Write a detailed technical explanation of how TCP congestion control works."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bank", choices=("percall", "native"), required=True)
    ap.add_argument("--budget-mb", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    model, tok = load(str(ROOT / args.model), lazy=True)
    if args.bank == "native":
        from rapidai_tools.native_moe import install_streaming_native
        stats = install_streaming_native(
            model, str(ROOT / args.model), cache_bytes=args.budget_mb << 20)
    else:
        from rapidai_tools.streamed_moe import install_streaming
        stats = install_streaming(
            model, str(ROOT / args.model), cache_bytes=args.budget_mb << 20)
    mx.eval(model.parameters())

    n = 0
    t0 = None
    b0 = h0 = m0 = None
    for r in stream_generate(model, tok, prompt=PROMPT,
                             max_tokens=args.max_tokens):
        if t0 is None:
            t0 = time.perf_counter()
            b0 = stats.bytes_read
            h0, m0 = stats.cache.hits, stats.cache.misses
        n += 1
    dt = time.perf_counter() - t0
    hits = stats.cache.hits - h0
    misses = stats.cache.misses - m0
    out = {
        "experiment": "phase1b native e2e A/B",
        "model": args.model,
        "bank": args.bank,
        "runs": [{
            "budget_mb": args.budget_mb,
            "tokens": n,
            "tok_s": round(n / dt, 3),
            "bytes_read_per_token": int((stats.bytes_read - b0) / max(n, 1)),
            "decode_cache_hit_rate": round(hits / max(hits + misses, 1), 4),
        }],
    }
    print(json.dumps(out["runs"][0]), flush=True)
    (ROOT / f"docs/experiments/data/phase1_bench_{args.tag}.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
