import numpy as np
from safetensors.numpy import save_file

from rapidai_tools.st_reader import STReader


def _mk(tmp_path):
    t = {
        "experts.w": np.arange(24, dtype=np.uint32).reshape(4, 3, 2),
        "scales": np.arange(12, dtype=np.float16).reshape(4, 3),
    }
    p = tmp_path / "t.safetensors"
    save_file(t, str(p))
    return p, t


def test_meta_shapes(tmp_path):
    p, t = _mk(tmp_path)
    r = STReader(str(p))
    assert r.tensors["experts.w"].shape == (4, 3, 2)
    assert r.tensors["scales"].dtype == "F16"


def test_read_row_matches_numpy(tmp_path):
    p, t = _mk(tmp_path)
    r = STReader(str(p))
    got = r.read_rows("experts.w", 2)
    np.testing.assert_array_equal(got, t["experts.w"][2])
    got_s = r.read_rows("scales", 3)
    np.testing.assert_array_equal(got_s, t["scales"][3])


def test_bytes_accounting(tmp_path):
    p, _ = _mk(tmp_path)
    r = STReader(str(p))
    r.read_rows("experts.w", 0)
    assert r.bytes_read == 3 * 2 * 4  # one row of uint32
