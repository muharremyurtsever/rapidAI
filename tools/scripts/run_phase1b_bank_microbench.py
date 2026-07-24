"""Phase 1b step 2 microbench: per-call compact bank vs persistent expert bank.

NEGATIVE-ish result (2026-07-24): the persistent bank measured 0.547 vs 0.561
ms/call here (1.03x) and 5.721 vs 5.090 tok/s end-to-end interleaved on
Qwen3-30B @ 6144 MB (1.124x) — below the pre-registered 1.15x win gate, so the
library implementation was reverted. This script keeps the full persistent-bank
implementation INLINE (SlotLRU + PersistentBankSwitchLinear) so the experiment
stays reproducible. Report: docs/experiments/phase1b-persistent-bank-report.md.

Within-process, INTERLEAVED A/B (host drift on this rig is up to 1.46x across
hours — see MEMORY.md benchmark rule 2). Real Qwen3-30B layer-0 up_proj
tensors, page-cache warmed. Unlike the step-1 fetch microbench, both variants
here run the FULL call path (bank assembly + gather_qmm + eval), because the
persistent bank changes the gather_qmm input (big resident bank + slot
indices) as well as the assembly.

Variants:
  percall    — shipped StreamedQuantizedSwitchLinear: SLRU of mx rows,
               per-call mx.stack compact bank, remapped indices.
  persistent — preallocated (slots, ...) mx banks per part, O(1) SlotLRU,
               batched scatter-write (bank[slots] = rows) of misses, then
               gather_qmm with slot indices; no per-call stack.

Slot capacity is matched to the SLRU byte budget so both variants hold the
same number of resident experts (same miss stream).
"""

import json
import sys
import time
from collections import OrderedDict
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
K_UNIQUE = 8  # experts per call (Qwen3 top-8)
CACHE_MB = 60  # forces a steady ~30% miss stream (like GPT-OSS at 8 GB)
GROUP, BITS = 64, 3
HIDDEN = 2048

_MX_DTYPES = {"F16": mx.float16, "BF16": mx.bfloat16, "U32": mx.uint32,
              "I8": mx.int8, "U8": mx.uint8, "F32": mx.float32}


class SlotLRU:
    """O(1) LRU over a fixed pool of persistent-bank slots."""

    def __init__(self, capacity_slots: int):
        self.capacity = capacity_slots
        self.slots: OrderedDict = OrderedDict()  # expert -> slot, MRU last
        self.free = list(range(capacity_slots))
        self.hits = self.misses = self.evictions = 0

    def get(self, expert: int):
        slot = self.slots.get(expert)
        if slot is None:
            self.misses += 1
            return None
        self.slots.move_to_end(expert)
        self.hits += 1
        return slot

    def assign(self, expert: int) -> int:
        if self.free:
            slot = self.free.pop()
        else:
            _, slot = self.slots.popitem(last=False)
            self.evictions += 1
        self.slots[expert] = slot
        return slot


class PersistentBankSwitchLinear:
    """QuantizedSwitchLinear on a preallocated resident bank + slot indices.

    (Was briefly in rapidai_tools.streamed_moe behind RAPIDAI_BANK=persistent;
    reverted after losing the pre-registered 1.15x e2e gate at 1.124x.)
    """

    def __init__(self, store: DiskExpertStore, group_size: int, bits: int,
                 capacity_slots: int, mode: str = "affine"):
        self.store = store
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.lru = SlotLRU(capacity_slots)
        self.banks = []
        for n in store.names:
            m = store.pool.by_name[f"{store.prefix}.{n}"].tensors[
                f"{store.prefix}.{n}"]
            self.banks.append(mx.zeros((capacity_slots,) + m.shape[1:],
                                       dtype=_MX_DTYPES[m.dtype]))
        mx.eval(*self.banks)

    def _read_row_np(self, expert: int):
        parts = []
        for n in self.store.names:
            full = f"{self.store.prefix}.{n}"
            parts.append(self.store.pool.by_name[full].read_rows(full, expert))
        return parts

    def __call__(self, x: mx.array, indices: mx.array,
                 sorted_indices: bool = False):
        idx_np = np.array(indices, copy=False).astype(np.int64)
        unique, inverse = np.unique(idx_np, return_inverse=True)
        slots = np.empty(len(unique), dtype=np.uint32)
        miss_pos = []
        for i, e in enumerate(unique):
            s = self.lru.get(int(e))
            if s is None:
                miss_pos.append(i)
            else:
                slots[i] = s
        if miss_pos:
            rows = [[] for _ in self.banks]
            miss_slots = np.empty(len(miss_pos), dtype=np.uint32)
            for j, i in enumerate(miss_pos):
                slot = self.lru.assign(int(unique[i]))
                slots[i] = slot
                miss_slots[j] = slot
                for p, r in enumerate(self._read_row_np(int(unique[i]))):
                    rows[p].append(r)
            sl = mx.array(miss_slots)
            for p, bank in enumerate(self.banks):
                upd = mx.array(np.stack(rows[p]))
                if bank.dtype == mx.bfloat16:
                    upd = upd.view(mx.bfloat16)
                bank[sl] = upd  # in-place scatter update
        slot_idx = mx.array(slots[inverse].reshape(idx_np.shape))
        banks = list(self.banks)
        lin_bias = banks.pop() if self.store.has_linear_bias else None
        weight, scales = banks[0], banks[1]
        biases = banks[2] if len(banks) > 2 else None
        out = mx.gather_qmm(x, weight, scales, biases, rhs_indices=slot_idx,
                            transpose=True, group_size=self.group_size,
                            bits=self.bits, mode=self.mode)
        if lin_bias is not None:
            out = out + mx.expand_dims(lin_bias[slot_idx], -2)
        return out


def run_percall(pool, calls, x):
    store = DiskExpertStore(pool, PREFIX, SLRUCache(CACHE_MB << 20))
    layer = StreamedQuantizedSwitchLinear(store, group_size=GROUP, bits=BITS)
    t0 = time.perf_counter()
    for idx in calls:
        mx.eval(layer(x, idx))
    dt = time.perf_counter() - t0
    c = store.cache
    return dt, c.hits / (c.hits + c.misses)


def run_persistent(pool, calls, x):
    store = DiskExpertStore(pool, PREFIX, SLRUCache(1))  # cache unused
    row_nb = sum(
        pool.by_name[f"{PREFIX}.{n}"].tensors[f"{PREFIX}.{n}"].nbytes
        // pool.by_name[f"{PREFIX}.{n}"].tensors[f"{PREFIX}.{n}"].shape[0]
        for n in store.names)
    capacity = min((CACHE_MB << 20) // row_nb, 128)
    layer = PersistentBankSwitchLinear(store, group_size=GROUP, bits=BITS,
                                       capacity_slots=capacity)
    t0 = time.perf_counter()
    for idx in calls:
        mx.eval(layer(x, idx))
    dt = time.perf_counter() - t0
    lru = layer.lru
    return dt, lru.hits / (lru.hits + lru.misses)


def main():
    pool = ReaderPool(MODEL)
    rng = np.random.default_rng(0)
    calls = [mx.array(np.sort(rng.choice(128, size=K_UNIQUE, replace=False))
                      .astype(np.uint32).reshape(1, 1, K_UNIQUE))
             for _ in range(N_CALLS)]
    x = mx.random.normal((1, 1, 1, HIDDEN)).astype(mx.bfloat16)
    mx.eval(x, *calls)
    warm = DiskExpertStore(pool, PREFIX, SLRUCache(1 << 62))
    for e in range(128):
        warm.fetch(e)  # page-cache warmup
    del warm
    variants = {"percall": run_percall, "persistent": run_persistent}
    results = {k: [] for k in variants}
    hits = {}
    for trial in range(3):
        for name, fn in variants.items():  # interleaved to control drift
            dt, hit = fn(pool, calls, x)
            ms = dt * 1000 / N_CALLS
            results[name].append(round(ms, 3))
            hits[name] = round(hit, 4)
            print(f"trial {trial} {name}: {ms:.3f} ms/call  hit={hit:.2f}",
                  flush=True)
    out = {
        "experiment": "phase1b persistent-bank microbench",
        "model": "qwen3-30b-a3b-3bit", "prefix": PREFIX,
        "n_calls": N_CALLS, "k_unique": K_UNIQUE, "cache_mb": CACHE_MB,
        "includes_gather_qmm": True,
        "hit_rate": hits,
        "ms_per_call": results,
        "median_ms_per_call": {k: sorted(v)[1] for k, v in results.items()},
    }
    (ROOT / "docs/experiments/data/phase1b_bank_microbench.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out["median_ms_per_call"], indent=2))


if __name__ == "__main__":
    main()
