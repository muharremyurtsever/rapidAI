"""Phase 1b native-port investigation: WHERE does the 0.55 ms/call go?

Decomposes the shipped StreamedQuantizedSwitchLinear call into phases and
measures four hypotheses on real Qwen3-30B layer-0 up_proj tensors
(page-cache warmed, 100%-hit cache so no disk I/O muddies orchestration):

  A. phase breakdown  — np.array(indices) sync, SLRU hit fetches, mx.stack
                        graph build, gather_qmm graph build, mx.eval.
  B. sync pattern     — N chained calls with ONE eval at the end vs eval per
                        call (how much is per-eval command-buffer overhead?).
  C. graph-free floor — persistent resident bank + PRECOMPUTED slot indices,
                        i.e. zero Python/np work and zero per-call sync:
                        144 chained gather_qmm in one graph, one eval.
                        This is the ceiling for a native (lazy-fetch) port.
  D. mx.compile       — is compiling the streamed call even legal? (fetch
                        happens at graph-build time and needs indices DATA;
                        tracer arrays have none — expect failure; verify.)

Results feed docs/experiments/phase1b-native-port-report.md.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from rapidai_tools.slru import SLRUCache  # noqa: E402
from rapidai_tools.streamed_moe import (  # noqa: E402
    DiskExpertStore,
    ReaderPool,
    StreamedQuantizedSwitchLinear,
)

ROOT = Path("/Volumes/x9/rapidAI")
MODEL = str(ROOT / "models/qwen3-30b-a3b-3bit")
PREFIX = "model.layers.0.mlp.switch_mlp.up_proj"
N_CALLS = 400
K_UNIQUE = 8
GROUP, BITS = 64, 3
HIDDEN = 2048
CALLS_PER_TOKEN = 144  # 48 layers x 3 projections on Qwen3-30B


def make_layer(pool, cache_bytes=1 << 62):
    store = DiskExpertStore(pool, PREFIX, SLRUCache(cache_bytes))
    return StreamedQuantizedSwitchLinear(store, group_size=GROUP, bits=BITS)


def phase_breakdown(pool, calls, x):
    """A: per-phase medians over N_CALLS calls, all cache hits."""
    layer = make_layer(pool)
    for e in range(128):
        layer.store.fetch(e)  # warm cache AND page cache
    t = {k: [] for k in ("np_indices", "fetch", "stack", "gather", "eval")}
    for idx in calls:
        t0 = time.perf_counter()
        idx_np = np.array(idx, copy=False).astype(np.int64)
        unique, inverse = np.unique(idx_np, return_inverse=True)
        t1 = time.perf_counter()
        fetched = [layer.store.fetch(int(e)) for e in unique]
        t2 = time.perf_counter()
        bank = [mx.stack([f[i] for f in fetched]) for i in range(len(fetched[0]))]
        t3 = time.perf_counter()
        compact_idx = mx.array(inverse.reshape(idx_np.shape).astype(np.uint32))
        out = mx.gather_qmm(x, bank[0], bank[1], bank[2],
                            rhs_indices=compact_idx, transpose=True,
                            group_size=GROUP, bits=BITS, mode="affine")
        t4 = time.perf_counter()
        mx.eval(out)
        t5 = time.perf_counter()
        for k, a, b in (("np_indices", t0, t1), ("fetch", t1, t2),
                        ("stack", t2, t3), ("gather", t3, t4), ("eval", t4, t5)):
            t[k].append((b - a) * 1000)
    return {k: round(float(np.median(v)), 4) for k, v in t.items()}


def sync_pattern(pool, calls, x):
    """B: eval per call vs one eval for the whole chain (same math)."""
    out = {}
    for mode in ("eval_per_call", "eval_once"):
        layer = make_layer(pool)
        for e in range(128):
            layer.store.fetch(e)
        t0 = time.perf_counter()
        y = x
        pend = []
        for idx in calls:
            o = layer(x, idx)
            if mode == "eval_per_call":
                mx.eval(o)
            else:
                pend.append(o)
        if pend:
            mx.eval(*pend)
        out[mode] = round((time.perf_counter() - t0) * 1000 / len(calls), 4)
    return out


def graph_free_floor(pool, x):
    """C: resident bank + precomputed slot indices; CALLS_PER_TOKEN chained
    gather_qmm ops, one eval — the native-port ceiling per 'token'."""
    store = DiskExpertStore(pool, PREFIX, SLRUCache(1 << 62))
    parts = [[] for _ in store.names]
    for e in range(128):
        for i, p in enumerate(store.fetch(e)):
            parts[i].append(p)
    banks = [mx.stack(p) for p in parts]
    mx.eval(*banks)
    rng = np.random.default_rng(1)
    idxs = [mx.array(np.sort(rng.choice(128, size=K_UNIQUE, replace=False))
                     .astype(np.uint32).reshape(1, 1, K_UNIQUE))
            for _ in range(CALLS_PER_TOKEN)]
    mx.eval(*idxs)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        outs = []
        for idx in idxs:
            outs.append(mx.gather_qmm(x, banks[0], banks[1], banks[2],
                                      rhs_indices=idx, transpose=True,
                                      group_size=GROUP, bits=BITS, mode="affine"))
        mx.eval(*outs)
        times.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(times))
    return {"ms_per_token_144_calls": round(med, 3),
            "ms_per_call": round(med / CALLS_PER_TOKEN, 4)}


def compile_check(pool, x):
    """D: does mx.compile accept the streamed call? (expect: no)"""
    layer = make_layer(pool)
    for e in range(128):
        layer.store.fetch(e)

    def fn(x, idx):
        return layer(x, idx)

    cfn = mx.compile(fn)
    idx = mx.array(np.arange(K_UNIQUE, dtype=np.uint32).reshape(1, 1, K_UNIQUE))
    try:
        mx.eval(cfn(x, idx))
        return {"compilable": True, "note": "compiled and evaluated"}
    except Exception as e:  # noqa: BLE001
        return {"compilable": False, "error": f"{type(e).__name__}: {e}"[:300]}


def main():
    pool = ReaderPool(MODEL)
    rng = np.random.default_rng(0)
    calls = [mx.array(np.sort(rng.choice(128, size=K_UNIQUE, replace=False))
                      .astype(np.uint32).reshape(1, 1, K_UNIQUE))
             for _ in range(N_CALLS)]
    x = mx.random.normal((1, 1, 1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x, *calls)
    results = {
        "experiment": "phase1b native-port investigation",
        "model": "qwen3-30b-a3b-3bit", "prefix": PREFIX,
        "A_phase_breakdown_ms": phase_breakdown(pool, calls, x),
        "B_sync_pattern_ms_per_call": sync_pattern(pool, calls, x),
        "C_graph_free_floor": graph_free_floor(pool, x),
        "D_mx_compile": compile_check(pool, x),
    }
    (ROOT / "docs/experiments/data/phase1b_native_investigation.json").write_text(
        json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
