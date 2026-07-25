"""Phase 3 / Bet B — predictability of t+lag routed experts.

Capture per-decode-step (hidden_state, routed_experts) for every MoE layer, then
compare a persistence baseline against a trained (ridge) predictor of token t+lag's
expert set, plus the miss-aware prefetch-coverage metric.

Usage:
  # GPT-OSS-120B (streamed, the model that matters)
  .venv/bin/python tools/scripts/run_phase3_predictability.py \
      --model models/gpt-oss-120b-mxfp4-q4 --tag gptoss120b \
      --stream --cache-mb 8192 --budget-mb 8192 \
      --n-prompts 4 --max-tokens 300

  # Qwen3-30B (resident, cross-check)
  .venv/bin/python tools/scripts/run_phase3_predictability.py \
      --model models/qwen3-30b-a3b-3bit --tag qwen3_30b \
      --budget-mb 6144 --n-prompts 5 --max-tokens 512
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load, generate

from rapidai_tools.bet_b_capture import (
    BOUNDARY, install_hidden_taps, build_arrays,
)
from rapidai_tools import bet_b_predict as bp

ROOT = Path("/Volumes/x9/rapidAI")

PROMPTS = [
    "Write a detailed essay about the history of the Ottoman Empire.",
    "Explain, step by step, how a B-tree database index works, with code examples.",
    "Write a short story about a lighthouse keeper on the Aegean coast.",
    "Derive the gradient of the softmax cross-entropy loss, showing every step.",
    "Translate the opening paragraph of Don Quixote into modern English and discuss its themes.",
    "Describe how TCP congestion control (slow start, AIMD, CUBIC) works in depth.",
]

LAGS = [1, 2, 3, 4]


def capture(args):
    n_layers_cfg = json.load(open(ROOT / args.model / "config.json"))["num_hidden_layers"]
    if args.stream:
        from rapidai_tools.streamed_moe import install_streaming
        model, tok = load(str(ROOT / args.model), lazy=True)
        install_streaming(model, str(ROOT / args.model), cache_bytes=args.cache_mb << 20)
        mx.eval(model.parameters())
    else:
        model, tok = load(str(ROOT / args.model))

    store: list = []
    n_taps = install_hidden_taps(model, store)
    print(f"taps installed: {n_taps} (config layers {n_layers_cfg})", flush=True)
    assert n_taps == n_layers_cfg, "tap count != MoE layer count"

    t0 = time.perf_counter()
    for i, p in enumerate(PROMPTS[: args.n_prompts]):
        store.append((BOUNDARY, None, None))
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True)
        generate(model, tok, prompt=prompt, max_tokens=args.max_tokens)
        mx.clear_cache()
        print(f"  gen {i+1}/{args.n_prompts} done  "
              f"({time.perf_counter()-t0:.0f}s, {len(store)} records)", flush=True)

    X, E = build_arrays(store, n_taps)
    print(f"captured X={X.shape} E={E.shape}", flush=True)
    traces = ROOT / "data/traces"
    traces.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(traces / f"betb_{args.tag}.npz", X=X, E=E)
    del model
    mx.clear_cache()
    return X, E


def analyze(args, X, E):
    n_experts = int(E.max()) + 1
    n_experts = max(n_experts, json.load(open(ROOT / args.model / "config.json")).get(
        "num_experts") or json.load(open(ROOT / args.model / "config.json")).get(
        "num_local_experts"))
    T, L, k = E.shape
    entry_bytes = bp.expert_entry_bytes(str(ROOT / args.model))
    budget_bytes = args.budget_mb << 20

    result = {
        "experiment": "phase3 bet-b predictability",
        "model": args.model,
        "tokens": int(T), "layers": int(L), "k": int(k),
        "n_experts": int(n_experts),
        "expert_entry_bytes": int(entry_bytes),
        "budget_mb": args.budget_mb,
        "train_frac": 0.7,
        "by_lag": {},
    }

    for lag in LAGS:
        pers_k = bp.persistence_recall(E, lag, 1)
        pers_2k = bp.persistence_recall(E, lag, 2)
        tr_k, tr_k_layers, eval_preds_k, eval_t = bp.trained_recall(
            X, E, n_experts, lag, 1)
        tr_2k, _, eval_preds_2k, _ = bp.trained_recall(X, E, n_experts, lag, 2)
        cov = bp.miss_coverage(
            E, X, n_experts, entry_bytes, budget_bytes, lag,
            eval_preds_k, eval_t, 1)
        entry = {
            "persistence_recall_at_k": round(pers_k, 4),
            "persistence_recall_at_2k": round(pers_2k, 4),
            "trained_recall_at_k": round(tr_k, 4),
            "trained_recall_at_2k": round(tr_2k, 4),
            "delta_pp_at_k": round((tr_k - pers_k) * 100, 2),
            "miss_coverage": cov,
        }
        if lag == 1:
            hy_k, _, _, _ = bp.trained_recall(
                X, E, n_experts, lag, 1, include_current_experts=True)
            entry["hybrid_recall_at_k"] = round(hy_k, 4)
            print(f"       hybrid(hidden+current-experts)@k={hy_k:.3f}", flush=True)
        result["by_lag"][str(lag)] = entry
        print(f"lag={lag}  pers@k={pers_k:.3f} trained@k={tr_k:.3f} "
              f"(Δ={100*(tr_k-pers_k):+.1f}pp)  pers@2k={pers_2k:.3f} "
              f"trained@2k={tr_2k:.3f}  "
              f"miss_cov pers={cov['persistence_miss_coverage']:.3f} "
              f"trained={cov['trained_miss_coverage']:.3f}", flush=True)

    g = result["by_lag"]["1"]
    result["gate"] = {
        "delta_pp_ge_15": g["delta_pp_at_k"] >= 15,
        "trained_recall_ge_60": g["trained_recall_at_k"] >= 0.60,
        "pass": bool(g["delta_pp_at_k"] >= 15 and g["trained_recall_at_k"] >= 0.60),
    }
    out = ROOT / f"docs/experiments/data/phase3_betb_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["gate"], indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--cache-mb", type=int, default=8192)
    ap.add_argument("--budget-mb", type=int, default=6144)
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if args.analyze_only:
        d = np.load(ROOT / f"data/traces/betb_{args.tag}.npz")
        X, E = d["X"], d["E"]
        print(f"loaded X={X.shape} E={E.shape}", flush=True)
    else:
        X, E = capture(args)
    analyze(args, X, E)


if __name__ == "__main__":
    main()
