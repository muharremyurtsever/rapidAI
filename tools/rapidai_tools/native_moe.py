"""Native (C++ extension) lazy expert-bank path — EXPERIMENTAL, not the default.

Phase 1b verdict (2026-07-25): LOSES the pre-registered e2e gate — 1.136x
(5.802 vs 5.109 tok/s interleaved medians on Qwen3-30B @ 6144 MB) against the
required 1.5x — so the library ships the Python per-call path and this module
is kept as a standalone experiment (installed via install_streaming_native,
never by rapidai_tools.streamed_moe). Report:
docs/experiments/phase1b-native-port-report.md.

Design: the shipped Python path must materialize the routing indices at graph-build
time (np.array(indices)) to fetch experts, which splits every token into
per-layer partial evals — measured 0.63 ms eval vs 0.03 ms Python work per
call (docs/experiments/phase1b-native-port-report.md). The native path defers
the fetch to graph-EVAL time: `_rapidai_bank.bank_fetch` is an MLX primitive
on the model's own stream. At eval it commits the pending Metal command
buffer with a completion handler that runs the slot LRU + pread-on-miss
straight into a persistent unified-memory bank, then gates later ops on an
MTLSharedEvent. It returns lazy [slot_indices, *bank_parts] that feed
mx.gather_qmm, so a whole token stays one MLX graph with no Python-side
synchronization.

The three projections of one MoE layer share the routing indices, so they
share ONE store + ONE fetch per layer (LayerBank): 48 fetch round-trips per
token instead of 144 on Qwen3-30B.

Build the extension first (see tools/native/):
  cd tools/native && PATH=$VENV/bin:$PATH cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$VENV/bin/python \
    && cmake --build build \
    && cp build/_rapidai_bank*.so ../rapidai_tools/
"""

import mlx.core as mx

try:
    from . import _rapidai_bank as _native
except ImportError:  # extension not built — install_streaming will refuse
    _native = None

_PART_NAMES = ("weight", "scales", "biases", "bias")


def native_available() -> bool:
    return _native is not None


def _probe_names(pool, prefix: str, layout: str):
    if layout == "stacked":
        return [n for n in _PART_NAMES if f"{prefix}.{n}" in pool]
    return [n for n in _PART_NAMES if prefix.format(e=0) + f".{n}" in pool]


def _n_experts(pool, prefix: str, layout: str, name0: str) -> int:
    if layout == "stacked":
        full = f"{prefix}.{name0}"
        return pool.by_name[full].tensors[full].shape[0]
    e = 0
    while prefix.format(e=e) + f".{name0}" in pool:
        e += 1
    return e


def _part_specs(pool, prefix: str, layout: str, names, n_experts: int):
    """Yield (name, spec, row_nbytes) PartSpec tuples for one projection."""
    for n in names:
        if layout == "stacked":
            full = f"{prefix}.{n}"
            reader = pool.by_name[full]
            m = reader.tensors[full]
            row_nb = m.nbytes // n_experts
            paths = [reader.path] * n_experts
            offsets = [m.start + e * row_nb for e in range(n_experts)]
            row_shape = list(m.shape[1:])
            dtype = m.dtype
        else:
            paths, offsets = [], []
            row_nb, row_shape, dtype = None, None, None
            for e in range(n_experts):
                full = prefix.format(e=e) + f".{n}"
                reader = pool.by_name[full]
                m = reader.tensors[full]
                if row_nb is None:
                    row_nb, row_shape, dtype = m.nbytes, list(m.shape), m.dtype
                elif m.nbytes != row_nb:
                    raise ValueError(f"ragged expert tensors for {prefix} {n}")
                paths.append(reader.path)
                offsets.append(m.start)
        yield n, (paths, offsets, row_nb, row_shape, dtype), row_nb


def make_layer_bank(pool, projections, capacity_bytes: int):
    """Build one shared NativeExpertStore for a whole MoE layer.

    `projections` is a list of (key, prefix, layout). Returns
    (LayerBank, {key: {"parts": {name: flat_output_index}, "has_bias": bool}}).
    """
    if _native is None:
        raise RuntimeError(
            "rapidai_tools._rapidai_bank extension is not built; "
            "see tools/native/ build instructions")
    first_key, first_prefix, first_layout = projections[0]
    names0 = _probe_names(pool, first_prefix, first_layout)
    n_experts = _n_experts(pool, first_prefix, first_layout, names0[0])
    specs = []
    layout_map = {}
    row_nbytes_total = 0
    for key, prefix, layout in projections:
        names = _probe_names(pool, prefix, layout)
        if _n_experts(pool, prefix, layout, names[0]) != n_experts:
            raise ValueError("projections disagree on expert count")
        part_pos = {}
        for n, spec, row_nb in _part_specs(pool, prefix, layout, names,
                                           n_experts):
            part_pos[n] = len(specs)
            specs.append(spec)
            row_nbytes_total += row_nb
        layout_map[key] = {"parts": part_pos, "has_bias": "bias" in part_pos}
    capacity_slots = max(1, min(capacity_bytes // row_nbytes_total, n_experts))
    store = _native.NativeExpertStore(
        n_experts=n_experts, capacity_slots=capacity_slots, parts=specs)
    return LayerBank(store, capacity_slots), layout_map


class LayerBank:
    """One shared fetch per MoE layer: the three projections receive the same
    routing indices object, so the first call fetches and the rest reuse the
    lazy result (identity-keyed; the reference keeps the id stable)."""

    def __init__(self, store, capacity_slots: int):
        self.store = store
        self.capacity = capacity_slots
        self._last_indices = None
        self._last_outs = None

    def fetch(self, idx: mx.array):
        if self._last_indices is not idx:
            self._last_indices = idx
            self._last_outs = _native.bank_fetch(
                mx.contiguous(idx.astype(mx.uint32)), self.store)
        return self._last_outs


class NativeBankSwitchLinear:
    """Drop-in for QuantizedSwitchLinear backed by a shared LayerBank.

    Calls wider than the slot capacity (prefill batches) fall back to a
    one-shot compact bank via the shipped Python path (`fallback`) — that
    path synchronizes, but prefill happens once and is compute-bound.
    """

    def __init__(self, bank: LayerBank, part_pos: dict, has_linear_bias: bool,
                 group_size: int, bits: int, mode: str, fallback):
        self.bank = bank
        self.part_pos = part_pos
        self.has_linear_bias = has_linear_bias
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.fallback = fallback

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False):
        # unique(indices) <= indices.size, so size <= capacity guarantees the
        # call fits in the bank; the check uses static shape info only (no sync).
        if indices.size > self.bank.capacity:
            return self.fallback(x, indices)
        outs = self.bank.fetch(indices)
        slot_idx = outs[0]
        weight = outs[1 + self.part_pos["weight"]]
        scales = outs[1 + self.part_pos["scales"]]
        biases = (outs[1 + self.part_pos["biases"]]
                  if "biases" in self.part_pos else None)
        out = mx.gather_qmm(
            x, weight, scales, biases,
            rhs_indices=slot_idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if self.has_linear_bias:
            lin_bias = outs[1 + self.part_pos["bias"]]
            out = out + mx.expand_dims(lin_bias[slot_idx], -2)
        return out


def install_streaming_native(model, model_dir: str, cache_bytes: int):
    """Experimental installer mirroring streamed_moe.install_streaming but
    with the native lazy bank. Returns a StreamStats-compatible object."""
    from .slru import SLRUCache
    from .streamed_moe import (
        DiskExpertStore,
        ReaderPool,
        StreamedQuantizedSwitchLinear,
        StreamStats,
    )

    pool = ReaderPool(model_dir)
    fallback_cache = SLRUCache(0)  # prefill fallback: read-through, no cache
    native_stores = []
    py_stores = []
    groups = []  # (sm, [(proj, prefix, layout)])
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        sm = getattr(mlp, "switch_mlp", None) or getattr(mlp, "experts", None)
        if sm is None or not hasattr(sm, "gate_proj"):
            continue
        projections = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            candidates = [
                (f"model.layers.{i}.mlp.switch_mlp.{proj}", "stacked"),
                (f"model.layers.{i}.mlp.experts.{proj}", "stacked"),
                (f"model.layers.{i}.mlp.experts.{{e}}.{proj}", "per_expert"),
            ]
            for prefix, layout in candidates:
                probe = prefix.format(e=0) if layout == "per_expert" else prefix
                if f"{probe}.weight" in pool:
                    projections.append((proj, prefix, layout))
                    break
            else:
                raise KeyError(f"no expert tensors found for layer {i} {proj}")
        groups.append((sm, projections))
    budget_share = cache_bytes // max(len(groups), 1)
    for sm, projections in groups:
        layer_bank, layout_map = make_layer_bank(pool, projections, budget_share)
        for proj, prefix, layout in projections:
            q = getattr(sm, proj)
            py_store = DiskExpertStore(pool, prefix, fallback_cache,
                                       layout=layout)
            fallback = StreamedQuantizedSwitchLinear(
                py_store, group_size=q.group_size, bits=q.bits,
                mode=getattr(q, "mode", "affine"))
            info = layout_map[proj]
            streamed = NativeBankSwitchLinear(
                layer_bank, info["parts"], info["has_bias"],
                group_size=q.group_size, bits=q.bits,
                mode=getattr(q, "mode", "affine"), fallback=fallback)
            setattr(sm, proj, streamed)
            py_stores.append(py_store)
        native_stores.append(layer_bank.store)
    stats = StreamStats(cache=NativeCacheView(native_stores, fallback_cache))
    stats.stores = native_stores + py_stores
    mx.clear_cache()
    return stats


class NativeCacheView:
    """SLRUCache-shaped stats facade over the native stores (+ fallback SLRU)."""

    def __init__(self, stores, fallback_cache):
        self._stores = stores
        self._fallback = fallback_cache

    @property
    def hits(self) -> int:
        return sum(s.hits for s in self._stores) + self._fallback.hits

    @property
    def misses(self) -> int:
        return sum(s.misses for s in self._stores) + self._fallback.misses

    @property
    def evictions(self) -> int:
        return sum(s.evictions for s in self._stores) + self._fallback.evictions

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "resident_bytes": None,  # fixed preallocated banks
        }
