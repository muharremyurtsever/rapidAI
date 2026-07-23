import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

from rapidai_tools.slru import SLRUCache
from rapidai_tools.streamed_moe import (
    DiskExpertStore,
    ReaderPool,
    StreamedQuantizedSwitchLinear,
)

GROUP = 32
BITS = 4
E, OUT, IN = 4, 8, 32
PACKED = IN * BITS // 32  # uint32 words per row


def _mk_quantized_bank(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.integers(0, 2**32, size=(E, OUT, PACKED), dtype=np.uint32)
    s = rng.normal(size=(E, OUT, IN // GROUP)).astype(np.float16)
    b = rng.normal(size=(E, OUT, IN // GROUP)).astype(np.float16)
    tensors = {"layer.weight": w, "layer.scales": s, "layer.biases": b}
    p = tmp_path / "bank.safetensors"
    save_file(tensors, str(p))
    return p, tensors


def test_fetch_returns_expert_rows_and_caches(tmp_path):
    p, t = _mk_quantized_bank(tmp_path)
    store = DiskExpertStore(ReaderPool(str(tmp_path)), "layer", SLRUCache(10**6))
    w, s, b = store.fetch(2)
    np.testing.assert_array_equal(np.array(w), t["layer.weight"][2])
    store.fetch(2)
    assert store.cache.stats()["hits"] == 1
    expected = (
        t["layer.weight"][2].nbytes
        + t["layer.scales"][2].nbytes
        + t["layer.biases"][2].nbytes
    )
    assert store.bytes_read == expected


def test_per_expert_layout(tmp_path):
    rng = np.random.default_rng(2)
    tensors = {}
    per_expert = {}
    for e in range(E):
        w = rng.integers(0, 2**32, size=(OUT, PACKED), dtype=np.uint32)
        tensors[f"experts.{e}.up.weight"] = w
        tensors[f"experts.{e}.up.scales"] = rng.normal(size=(OUT, 1)).astype(np.float16)
        tensors[f"experts.{e}.up.biases"] = rng.normal(size=(OUT, 1)).astype(np.float16)
        per_expert[e] = w
    save_file(tensors, str(tmp_path / "pe.safetensors"))
    store = DiskExpertStore(
        ReaderPool(str(tmp_path)), "experts.{e}.up", SLRUCache(10**6),
        layout="per_expert")
    w, s, b = store.fetch(3)
    np.testing.assert_array_equal(np.array(w), per_expert[3])


def test_streamed_matches_full_gather_qmm(tmp_path):
    p, t = _mk_quantized_bank(tmp_path)
    store = DiskExpertStore(ReaderPool(str(tmp_path)), "layer", SLRUCache(10**6))
    layer = StreamedQuantizedSwitchLinear(store, group_size=GROUP, bits=BITS)

    x = mx.array(np.random.default_rng(1).normal(size=(1, 1, 1, IN)).astype(np.float16))
    idx = mx.array(np.array([[[1, 3, 1]]], dtype=np.uint32))  # duplicate expert on purpose

    ref = mx.gather_qmm(
        x,
        mx.array(t["layer.weight"]),
        mx.array(t["layer.scales"]),
        mx.array(t["layer.biases"]),
        rhs_indices=idx,
        transpose=True,
        group_size=GROUP,
        bits=BITS,
    )
    got = layer(x, idx)
    assert mx.allclose(got, ref, atol=1e-3).item()
