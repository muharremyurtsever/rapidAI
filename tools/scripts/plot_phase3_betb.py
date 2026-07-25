"""Plot Bet B predictability: persistence vs trained recall@k across lags."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Volumes/x9/rapidAI")


def plot(tag):
    d = json.load(open(ROOT / f"docs/experiments/data/phase3_betb_{tag}.json"))
    lags = sorted(int(x) for x in d["by_lag"])
    pers = [d["by_lag"][str(l)]["persistence_recall_at_k"] for l in lags]
    trained = [d["by_lag"][str(l)]["trained_recall_at_k"] for l in lags]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lags, pers, "o-", label="persistence (free)")
    ax.plot(lags, trained, "s-", label="trained (hidden-state ridge)")
    ax.axhline(0.60, color="red", ls="--", label="gate abs (0.60)")
    ax.set_xlabel("prefetch horizon lag (tokens ahead)")
    ax.set_ylabel("recall@k of t+lag experts")
    ax.set_title(f"Bet B predictability — {d['model'].split('/')[-1]} "
                 f"({d['tokens']} tok)")
    ax.set_ylim(0, 1)
    ax.set_xticks(lags)
    ax.legend()
    fig.tight_layout()
    out = ROOT / f"docs/experiments/data/phase3_betb_{tag}.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        plot(tag)
