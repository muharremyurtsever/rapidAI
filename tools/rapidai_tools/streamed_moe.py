"""Disk-backed MoE expert bank for MLX models (equation factor: M_miss).

DiskExpertStore fetches one expert's quantized rows (weight/scales/biases)
via os.pread through a shared SLRU cache. StreamedQuantizedSwitchLinear is a
drop-in replacement for mlx_lm's QuantizedSwitchLinear: on each call it
assembles a compact bank of just the unique experts referenced by `indices`,
remaps the indices, and runs the same mx.gather_qmm — identical math, no
resident expert weights.
"""

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .slru import SLRUCache
from .st_reader import STReader


class DiskExpertStore:
    """Per-tensor-prefix expert fetcher: reads {prefix}.{weight,scales,biases} rows."""

    def __init__(self, st_path: str, prefix: str, cache: SLRUCache,
                 names=("weight", "scales", "biases")):
        self.reader = STReader(st_path)
        self.prefix = prefix
        self.names = [n for n in names if f"{prefix}.{n}" in self.reader.tensors]
        self.cache = cache
        self.bytes_read = 0

    def _to_mx(self, name_full: str, arr: np.ndarray) -> mx.array:
        if self.reader.tensors[name_full].dtype == "BF16":
            return mx.array(arr).view(mx.bfloat16)
        return mx.array(arr)

    def fetch(self, expert: int):
        key = (self.prefix, expert)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        parts = []
        nbytes = 0
        for n in self.names:
            full = f"{self.prefix}.{n}"
            arr = self.reader.read_rows(full, expert)
            nbytes += arr.nbytes
            parts.append(self._to_mx(full, arr))
        self.bytes_read += nbytes
        value = tuple(parts)
        self.cache.put(key, value, nbytes)
        return value


class StreamedQuantizedSwitchLinear:
    """Drop-in for QuantizedSwitchLinear backed by a DiskExpertStore."""

    def __init__(self, store: DiskExpertStore, group_size: int, bits: int,
                 mode: str = "affine"):
        self.store = store
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False):
        idx_np = np.array(indices, copy=False).astype(np.int64)
        unique, inverse = np.unique(idx_np, return_inverse=True)
        fetched = [self.store.fetch(int(e)) for e in unique]
        bank = [mx.stack([f[i] for f in fetched]) for i in range(len(fetched[0]))]
        weight, scales = bank[0], bank[1]
        biases = bank[2] if len(bank) > 2 else None
        compact_idx = mx.array(inverse.reshape(idx_np.shape).astype(np.uint32))
        return mx.gather_qmm(
            x, weight, scales, biases,
            rhs_indices=compact_idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


@dataclass
class StreamStats:
    stores: list = field(default_factory=list)
    cache: SLRUCache = None

    @property
    def bytes_read(self) -> int:
        return sum(s.bytes_read for s in self.stores)

    @property
    def expert_fetches(self) -> int:
        return self.cache.hits + self.cache.misses if self.cache else 0


def install_streaming(model, model_dir: str, cache_bytes: int) -> StreamStats:
    """Swap every MoE block's expert bank for disk-backed streamed versions."""
    import glob
    import os

    cache = SLRUCache(cache_bytes)
    stats = StreamStats(cache=cache)
    name_to_file: dict = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        r = STReader(f)
        for n in r.tensors:
            name_to_file[n] = f
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            continue
        sm = mlp.switch_mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            q = getattr(sm, proj)
            prefix = f"model.layers.{i}.mlp.switch_mlp.{proj}"
            if f"{prefix}.weight" not in name_to_file:
                raise KeyError(f"tensor {prefix}.weight not found in safetensors")
            store = DiskExpertStore(name_to_file[f"{prefix}.weight"], prefix, cache)
            streamed = StreamedQuantizedSwitchLinear(
                store, group_size=q.group_size, bits=q.bits,
                mode=getattr(q, "mode", "affine"))
            # drop resident expert tensors, then swap the module
            setattr(sm, proj, streamed)
            stats.stores.append(store)
    mx.clear_cache()
    return stats
