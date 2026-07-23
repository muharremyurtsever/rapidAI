"""Minimal safetensors random-access reader using os.pread.

Header format: 8-byte little-endian u64 header length, then JSON mapping
tensor name -> {dtype, shape, data_offsets: [begin, end]} relative to the
byte after the header. Reading rows with pread (not mmap) is the fast path
on Apple Silicon per experiment 0.2.
"""

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

_DTYPES = {
    "F16": np.float16,
    "BF16": np.uint16,  # raw bits; caller reinterprets
    "U32": np.uint32,
    "I8": np.int8,
    "U8": np.uint8,
    "F32": np.float32,
}


@dataclass
class TensorMeta:
    shape: tuple
    dtype: str
    start: int  # absolute file offset of tensor data
    nbytes: int


class STReader:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        header_len = struct.unpack("<Q", os.pread(self.fd, 8, 0))[0]
        header = json.loads(os.pread(self.fd, header_len, 8))
        data_base = 8 + header_len
        self.tensors: dict = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            b, e = info["data_offsets"]
            self.tensors[name] = TensorMeta(
                shape=tuple(info["shape"]),
                dtype=info["dtype"],
                start=data_base + b,
                nbytes=e - b,
            )
        self.bytes_read = 0

    def read_full(self, name: str) -> np.ndarray:
        m = self.tensors[name]
        raw = os.pread(self.fd, m.nbytes, m.start)
        self.bytes_read += m.nbytes
        return np.frombuffer(raw, dtype=_DTYPES[m.dtype]).reshape(m.shape)

    def read_rows(self, name: str, row: int) -> np.ndarray:
        m = self.tensors[name]
        n_rows = m.shape[0]
        row_bytes = m.nbytes // n_rows
        raw = os.pread(self.fd, row_bytes, m.start + row * row_bytes)
        self.bytes_read += row_bytes
        arr = np.frombuffer(raw, dtype=_DTYPES[m.dtype])
        return arr.reshape(m.shape[1:])

    def __del__(self):
        try:
            os.close(self.fd)
        except (OSError, AttributeError):
            pass
