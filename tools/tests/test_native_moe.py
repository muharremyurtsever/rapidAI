"""Native lazy-bank path (RAPIDAI_BANK=native) vs mx.gather_qmm reference."""

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from rapidai_tools.slru import SLRUCache
from rapidai_tools.streamed_moe import (
    DiskExpertStore,
    ReaderPool,
    StreamedQuantizedSwitchLinear,
)
from rapidai_tools.native_moe import (
    NativeBankSwitchLinear,
    make_layer_bank,
    native_available,
)

pytestmark = pytest.mark.skipif(
    not native_available(), reason="_rapidai_bank extension not built")

GROUP = 32
BITS = 4
E, OUT, IN = 4, 8, 32
PACKED = IN * BITS // 32
ROW_NB = (E * OUT * PACKED * 4 + 2 * E * OUT * (IN // GROUP) * 2) // E


def _mk_bank(tmp_path, with_linear_bias=False):
    rng = np.random.default_rng(0)
    tensors = {
        "layer.weight": rng.integers(0, 2**32, size=(E, OUT, PACKED),
                                     dtype=np.uint32),
        "layer.scales": rng.normal(size=(E, OUT, IN // GROUP)).astype(np.float16),
        "layer.biases": rng.normal(size=(E, OUT, IN // GROUP)).astype(np.float16),
    }
    if with_linear_bias:
        tensors["layer.bias"] = rng.normal(size=(E, OUT)).astype(np.float16)
    save_file(tensors, str(tmp_path / "bank.safetensors"))
    return tensors


def _mk_layer(tmp_path, capacity_bytes=10**9, with_linear_bias=False):
    t = _mk_bank(tmp_path, with_linear_bias)
    pool = ReaderPool(str(tmp_path))
    bank, layout_map = make_layer_bank(
        pool, [("p", "layer", "stacked")], capacity_bytes)
    fallback = StreamedQuantizedSwitchLinear(
        DiskExpertStore(pool, "layer", SLRUCache(0)),
        group_size=GROUP, bits=BITS)
    info = layout_map["p"]
    layer = NativeBankSwitchLinear(bank, info["parts"], info["has_bias"],
                                   group_size=GROUP, bits=BITS, mode="affine",
                                   fallback=fallback)
    return layer, t


def _reference(t, x, idx):
    ref = mx.gather_qmm(
        x, mx.array(t["layer.weight"]), mx.array(t["layer.scales"]),
        mx.array(t["layer.biases"]), rhs_indices=idx, transpose=True,
        group_size=GROUP, bits=BITS)
    if "layer.bias" in t:
        ref = ref + mx.expand_dims(mx.array(t["layer.bias"])[idx], -2)
    return ref


def _x():
    return mx.array(
        np.random.default_rng(1).normal(size=(1, 1, 1, IN)).astype(np.float16))


def test_native_matches_reference(tmp_path):
    layer, t = _mk_layer(tmp_path)
    x = _x()
    idx = mx.array(np.array([[[1, 3, 1]]], dtype=np.uint32))  # dup on purpose
    assert mx.allclose(layer(x, idx), _reference(t, x, idx), atol=1e-3).item()


def test_native_linear_bias(tmp_path):
    layer, t = _mk_layer(tmp_path, with_linear_bias=True)
    x = _x()
    idx = mx.array(np.array([[[0, 2]]], dtype=np.uint32))
    assert mx.allclose(layer(x, idx), _reference(t, x, idx), atol=1e-3).item()


def test_native_eviction_chain_matches(tmp_path):
    # capacity 2 slots (< E=4): every call evicts; math must still match.
    layer, t = _mk_layer(tmp_path, capacity_bytes=2 * ROW_NB)
    assert layer.bank.capacity == 2
    x = _x()
    seq = [[0, 1], [2, 3], [1, 0], [3, 1], [2, 2]]
    for pair in seq:
        idx = mx.array(np.array([[pair]], dtype=np.uint32))
        assert mx.allclose(layer(x, idx), _reference(t, x, idx),
                           atol=1e-3).item(), pair
    assert layer.bank.store.evictions > 0


def test_native_slot_reuse_no_refetch(tmp_path):
    layer, t = _mk_layer(tmp_path)
    x = _x()
    idx = mx.array(np.array([[[1, 2]]], dtype=np.uint32))
    mx.eval(layer(x, idx))
    b0 = layer.bank.store.bytes_read
    idx2 = mx.array(np.array([[[1, 2]]], dtype=np.uint32))
    mx.eval(layer(x, idx2))
    assert layer.bank.store.bytes_read == b0
    assert layer.bank.store.hits >= 2


def test_native_oversize_call_falls_back(tmp_path):
    layer, t = _mk_layer(tmp_path, capacity_bytes=2 * ROW_NB)
    x = mx.array(np.random.default_rng(3).normal(
        size=(1, 3, 1, IN)).astype(np.float16))
    idx = mx.array(np.array([[[0], [1], [2]]], dtype=np.uint32))  # size 3 > 2
    got = layer(x, idx)
    ref = _reference(t, x, idx)
    assert mx.allclose(got, ref, atol=1e-3).item()
    assert layer.bank.store.misses == 0  # native bank untouched


def test_native_lazy_chain_single_eval(tmp_path):
    # No sync between calls: chain several calls, eval once at the end.
    layer, t = _mk_layer(tmp_path)
    x = _x()
    idxs = [mx.array(np.array([[[a, b]]], dtype=np.uint32))
            for a, b in [(0, 1), (2, 3), (3, 0), (1, 2)]]
    outs = [layer(x, idx) for idx in idxs]
    refs = [_reference(t, x, idx) for idx in idxs]
    mx.eval(*outs)
    for got, ref in zip(outs, refs):
        assert mx.allclose(got, ref, atol=1e-3).item()


def test_layer_bank_shared_fetch(tmp_path):
    # Two projections in one group share one store and one fetch per indices
    # object (same-slot mapping across projections).
    rng = np.random.default_rng(4)
    tensors = {}
    for p in ("up", "gate"):
        tensors[f"{p}.weight"] = rng.integers(0, 2**32, size=(E, OUT, PACKED),
                                              dtype=np.uint32)
        tensors[f"{p}.scales"] = rng.normal(
            size=(E, OUT, IN // GROUP)).astype(np.float16)
        tensors[f"{p}.biases"] = rng.normal(
            size=(E, OUT, IN // GROUP)).astype(np.float16)
    save_file(tensors, str(tmp_path / "two.safetensors"))
    pool = ReaderPool(str(tmp_path))
    bank, layout_map = make_layer_bank(
        pool, [("up", "up", "stacked"), ("gate", "gate", "stacked")], 10**9)
    layers = {
        k: NativeBankSwitchLinear(bank, layout_map[k]["parts"],
                                  layout_map[k]["has_bias"], group_size=GROUP,
                                  bits=BITS, mode="affine", fallback=None)
        for k in ("up", "gate")
    }
    x = _x()
    idx = mx.array(np.array([[[3, 0]]], dtype=np.uint32))
    outs = {k: layers[k](x, idx) for k in layers}
    mx.eval(*outs.values())
    assert bank.store.hits + bank.store.misses == 2  # one fetch, two experts
    for k in layers:
        ref = mx.gather_qmm(
            x, mx.array(tensors[f"{k}.weight"]), mx.array(tensors[f"{k}.scales"]),
            mx.array(tensors[f"{k}.biases"]), rhs_indices=idx, transpose=True,
            group_size=GROUP, bits=BITS)
        assert mx.allclose(outs[k], ref, atol=1e-3).item(), k


def test_native_per_expert_layout(tmp_path):
    rng = np.random.default_rng(2)
    tensors = {}
    for e in range(E):
        tensors[f"experts.{e}.up.weight"] = rng.integers(
            0, 2**32, size=(OUT, PACKED), dtype=np.uint32)
        tensors[f"experts.{e}.up.scales"] = rng.normal(
            size=(OUT, IN // GROUP)).astype(np.float16)
        tensors[f"experts.{e}.up.biases"] = rng.normal(
            size=(OUT, IN // GROUP)).astype(np.float16)
    save_file(tensors, str(tmp_path / "pe.safetensors"))
    pool = ReaderPool(str(tmp_path))
    bank, layout_map = make_layer_bank(
        pool, [("up", "experts.{e}.up", "per_expert")], 10**9)
    assert bank.store.n_experts == E
    layer = NativeBankSwitchLinear(
        bank, layout_map["up"]["parts"], layout_map["up"]["has_bias"],
        group_size=GROUP, bits=BITS, mode="affine", fallback=None)
    x = _x()
    idx = mx.array(np.array([[[3, 0]]], dtype=np.uint32))
    w = np.stack([tensors[f"experts.{e}.up.weight"] for e in range(E)])
    s = np.stack([tensors[f"experts.{e}.up.scales"] for e in range(E)])
    b = np.stack([tensors[f"experts.{e}.up.biases"] for e in range(E)])
    ref = mx.gather_qmm(x, mx.array(w), mx.array(s), mx.array(b),
                        rhs_indices=idx, transpose=True,
                        group_size=GROUP, bits=BITS)
    assert mx.allclose(layer(x, idx), ref, atol=1e-3).item()
