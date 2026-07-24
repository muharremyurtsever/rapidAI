"""Phase 1b step 1 microbench: per-expert fetch loop vs batched-fetch variants.

Within-process, interleaved A/B on real Qwen3-30B layer-0 tensors. After a
page-cache warmup, preads are CPU-cheap, so timings isolate the Python/MLX
orchestration overhead of assembling the compact expert bank — the exact
target of the "batched fetch" optimization — independent of host/SSD state
drift (which was measured at up to 1.46x across a single afternoon and
swamps end-to-end tok/s comparisons).

Variants:
  old     — shipped path: per-expert fetch (pread + frombuffer + mx.array per
            part), bank = mx.stack of cached mx rows.
  many    — batched reads into ONE bytearray per part, ONE np.frombuffer +
            mx.array per part; cache entries are lazy mx slices; bank still
            mx.stack per part.
  banked  — like many, but bank = concat(stack(hits), batched_misses); miss
            slices never enter the call's graph.
  npbank  — cache numpy rows, assemble bank with np.stack, ONE mx.array
            conversion per part per call.

Result (2026-07-24): old wins at steady state. Preads are only ~0.16 ms/call;
every batched variant pays extra full copies of miss bytes and loses.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from rapidai_tools.slru import SLRUCache  # noqa: E402
from rapidai_tools.st_reader import _DTYPES  # noqa: E402
from rapidai_tools.streamed_moe import DiskExpertStore, ReaderPool  # noqa: E402

ROOT = Path("/Volumes/x9/rapidAI")
MODEL = str(ROOT / "models/qwen3-30b-a3b-3bit")
PREFIX = "model.layers.0.mlp.switch_mlp.up_proj"
N_CALLS = 400
K_UNIQUE = 8  # experts per call
CACHE_MB = 60  # forces a steady ~30% miss stream (like GPT-OSS at 8 GB)


def preads_into(reader, meta, expert_ids, buf):
    row_bytes = meta.nbytes // meta.shape[0]
    mv = memoryview(buf)
    for j, e in enumerate(expert_ids):
        os.preadv(reader.fd, [mv[j * row_bytes:(j + 1) * row_bytes]],
                  meta.start + e * row_bytes)


def read_part_batched(pool, name, expert_ids):
    full = f"{PREFIX}.{name}"
    reader = pool.by_name[full]
    meta = reader.tensors[full]
    row_bytes = meta.nbytes // meta.shape[0]
    buf = bytearray(row_bytes * len(expert_ids))
    preads_into(reader, meta, expert_ids, buf)
    arr = np.frombuffer(buf, dtype=_DTYPES[meta.dtype])
    arr = arr.reshape((len(expert_ids),) + meta.shape[1:])
    out = mx.array(arr)
    if meta.dtype == "BF16":
        out = out.view(mx.bfloat16)
    return out


def run_old(pool, names, calls):
    st = DiskExpertStore(pool, PREFIX, SLRUCache(CACHE_MB << 20))
    t0 = time.perf_counter()
    for ids in calls:
        fetched = [st.fetch(e) for e in ids]
        bank = [mx.stack([f[i] for f in fetched]) for i in range(len(names))]
        mx.eval(*bank)
    return time.perf_counter() - t0, st.cache


def run_many(pool, names, calls):
    cache = SLRUCache(CACHE_MB << 20)
    t0 = time.perf_counter()
    for ids in calls:
        out = [None] * len(ids)
        missing = []
        for i, e in enumerate(ids):
            v = cache.get(e)
            if v is None:
                missing.append((i, e))
            else:
                out[i] = v
        if missing:
            miss_ids = [e for _, e in missing]
            batched = [read_part_batched(pool, n, miss_ids) for n in names]
            nb = sum(b.nbytes for b in batched) // len(miss_ids)
            for j, (pos, e) in enumerate(missing):
                v = tuple(b[j] for b in batched)
                cache.put(e, v, nb)
                out[pos] = v
        bank = [mx.stack([f[i] for f in out]) for i in range(len(names))]
        mx.eval(*bank)
    return time.perf_counter() - t0, cache


def run_banked(pool, names, calls):
    cache = SLRUCache(CACHE_MB << 20)
    t0 = time.perf_counter()
    for ids in calls:
        hit_vals, missing = [], []
        for i, e in enumerate(ids):
            v = cache.get(e)
            if v is None:
                missing.append((i, e))
            else:
                hit_vals.append(v)
        if missing:
            miss_ids = [e for _, e in missing]
            batched = [read_part_batched(pool, n, miss_ids) for n in names]
            nb = sum(b.nbytes for b in batched) // len(miss_ids)
            for j, (_, e) in enumerate(missing):
                cache.put(e, tuple(b[j] for b in batched), nb)
            if hit_vals:
                bank = [mx.concatenate(
                    [mx.stack([v[i] for v in hit_vals]), batched[i]])
                    for i in range(len(names))]
            else:
                bank = batched
        else:
            bank = [mx.stack([v[i] for v in hit_vals])
                    for i in range(len(names))]
        mx.eval(*bank)
    return time.perf_counter() - t0, cache


def run_npbank(pool, names, calls):
    cache = SLRUCache(CACHE_MB << 20)
    metas = {n: pool.by_name[f"{PREFIX}.{n}"].tensors[f"{PREFIX}.{n}"]
             for n in names}
    readers = {n: pool.by_name[f"{PREFIX}.{n}"] for n in names}
    t0 = time.perf_counter()
    for ids in calls:
        fetched = []
        for e in ids:
            v = cache.get(e)
            if v is None:
                parts, nb = [], 0
                for n in names:
                    m = metas[n]
                    rb = m.nbytes // m.shape[0]
                    raw = os.pread(readers[n].fd, rb, m.start + e * rb)
                    parts.append(np.frombuffer(raw, dtype=_DTYPES[m.dtype])
                                 .reshape(m.shape[1:]))
                    nb += rb
                v = tuple(parts)
                cache.put(e, v, nb)
            fetched.append(v)
        bank = []
        for i, n in enumerate(names):
            b = mx.array(np.stack([f[i] for f in fetched]))
            if metas[n].dtype == "BF16":
                b = b.view(mx.bfloat16)
            bank.append(b)
        mx.eval(*bank)
    return time.perf_counter() - t0, cache


def main():
    pool = ReaderPool(MODEL)
    rng = np.random.default_rng(0)
    calls = [sorted(int(e) for e in rng.choice(128, size=K_UNIQUE,
                                               replace=False))
             for _ in range(N_CALLS)]
    warm = DiskExpertStore(pool, PREFIX, SLRUCache(1 << 62))
    names = warm.names
    for e in range(128):
        warm.fetch(e)  # page-cache warmup
    variants = {"old": run_old, "many": run_many,
                "banked": run_banked, "npbank": run_npbank}
    results = {k: [] for k in variants}
    for trial in range(3):
        for name, fn in variants.items():  # interleaved to control drift
            dt, cache = fn(pool, names, calls)
            ms = dt * 1000 / N_CALLS
            hit = cache.hits / (cache.hits + cache.misses)
            results[name].append(round(ms, 3))
            print(f"trial {trial} {name}: {ms:.3f} ms/call  hit={hit:.2f}",
                  flush=True)
    out = {
        "experiment": "phase1b fetch microbench",
        "model": "qwen3-30b-a3b-3bit", "prefix": PREFIX,
        "n_calls": N_CALLS, "k_unique": K_UNIQUE, "cache_mb": CACHE_MB,
        "ms_per_call": results,
        "median_ms_per_call": {k: sorted(v)[1] for k, v in results.items()},
    }
    (ROOT / "docs/experiments/data/phase1b_fetch_microbench.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out["median_ms_per_call"], indent=2))


if __name__ == "__main__":
    main()
