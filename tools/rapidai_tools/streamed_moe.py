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


class ReaderPool:
    """Shared open-file STReader pool + tensor-name -> reader catalog."""

    def __init__(self, model_dir: str):
        import glob
        import os as _os

        self.by_name: dict = {}
        for f in sorted(glob.glob(_os.path.join(model_dir, "*.safetensors"))):
            r = STReader(f)
            for n in r.tensors:
                self.by_name[n] = r

    def __contains__(self, name: str) -> bool:
        return name in self.by_name


class DiskExpertStore:
    """Expert fetcher supporting both on-disk layouts:

    - "stacked": one tensor per projection, expert = row of axis 0
      ({prefix}.{part}, e.g. model.layers.3.mlp.switch_mlp.up_proj.weight)
    - "per_expert": one tensor per expert
      ({prefix} with '{e}' placeholder, e.g. model.layers.3.mlp.experts.{e}.up_proj.weight)
    """

    def __init__(self, pool, prefix: str, cache: SLRUCache, layout: str = "stacked",
                 names=("weight", "scales", "biases")):
        self.pool = pool
        self.prefix = prefix
        self.layout = layout
        probe = [n for n in names]
        if layout == "stacked":
            self.names = [n for n in probe if f"{prefix}.{n}" in pool]
        else:
            self.names = [n for n in probe if prefix.format(e=0) + f".{n}" in pool]
        self.cache = cache
        self.bytes_read = 0

    def _to_mx(self, reader: STReader, name_full: str, arr: np.ndarray) -> mx.array:
        if reader.tensors[name_full].dtype == "BF16":
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
            if self.layout == "stacked":
                full = f"{self.prefix}.{n}"
                reader = self.pool.by_name[full]
                arr = reader.read_rows(full, expert)
            else:
                full = self.prefix.format(e=expert) + f".{n}"
                reader = self.pool.by_name[full]
                arr = reader.read_full(full)
            nbytes += arr.nbytes
            parts.append(self._to_mx(reader, full, arr))
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
    """Swap every MoE block's expert bank for disk-backed streamed versions.

    Detects the on-disk layout per projection: stacked switch_mlp tensors
    (Qwen3 MLX exports) or per-expert tensors (OLMoE MLX exports).
    """
    cache = SLRUCache(cache_bytes)
    stats = StreamStats(cache=cache)
    pool = ReaderPool(model_dir)
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            continue
        sm = mlp.switch_mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            q = getattr(sm, proj)
            stacked = f"model.layers.{i}.mlp.switch_mlp.{proj}"
            per_expert = f"model.layers.{i}.mlp.experts.{{e}}.{proj}"
            if f"{stacked}.weight" in pool:
                store = DiskExpertStore(pool, stacked, cache, layout="stacked")
            elif per_expert.format(e=0) + ".weight" in pool:
                store = DiskExpertStore(pool, per_expert, cache, layout="per_expert")
            else:
                raise KeyError(f"no expert tensors found for layer {i} {proj}")
            streamed = StreamedQuantizedSwitchLinear(
                store, group_size=q.group_size, bits=q.bits,
                mode=getattr(q, "mode", "affine"))
            setattr(sm, proj, streamed)
            stats.stores.append(store)
    mx.clear_cache()
    return stats
