"""Phase 1a gate: streamed OLMoE must reproduce resident OLMoE greedy tokens exactly."""

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate

from rapidai_tools.streamed_moe import install_streaming

ROOT = Path("/Volumes/x9/rapidAI")
MODEL = str(ROOT / "models/olmoe-1b-7b-4bit")
PROMPT = "Explain why the sky is blue in exactly three sentences."
N = 64


def greedy(model, tok):
    return generate(model, tok, prompt=PROMPT, max_tokens=N)


def main():
    model, tok = load(MODEL)
    reference = greedy(model, tok)
    del model
    mx.clear_cache()

    model2, tok2 = load(MODEL)
    stats = install_streaming(model2, MODEL, cache_bytes=512 * 1024 * 1024)
    streamed = greedy(model2, tok2)

    ok = reference == streamed
    out = {
        "experiment": "phase1 correctness gate",
        "model": "olmoe-1b-7b-4bit",
        "match": ok,
        "bytes_read": stats.bytes_read,
        "cache": stats.cache.stats(),
        "reference": reference,
        "streamed": streamed,
    }
    (ROOT / "docs/experiments/data/phase1_correctness_olmoe.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps({k: v for k, v in out.items() if k not in ("reference", "streamed")}, indent=2))
    assert ok, "STREAMED OUTPUT DIVERGED"


if __name__ == "__main__":
    main()
