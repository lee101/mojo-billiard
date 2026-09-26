"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Buffers cross the C ABI as 64-bit
addresses, so the argtypes below must stay ``c_int64``; ``c_int`` truncates and
segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-billiard.so"

_I64 = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.bl_bisect_left.restype = None
    lib.bl_bisect_left.argtypes = [_I64, _I64, _I64, _I64]
    lib.bl_insert_sorted.restype = ctypes.c_int64
    lib.bl_insert_sorted.argtypes = [_I64, _I64, _I64, _I64]
    lib.bl_remove_sorted.restype = ctypes.c_int64
    lib.bl_remove_sorted.argtypes = [_I64, _I64, _I64]
    lib.bl_prefix_sum.restype = None
    lib.bl_prefix_sum.argtypes = [_I64, _I64, _I64]
    lib.bl_merge_intervals.restype = ctypes.c_int64
    lib.bl_merge_intervals.argtypes = [_I64, _I64, _I64, _I64, _I64]
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def bisect_left(lengths, value: int) -> int:
    """`bisect.bisect_left` over a sorted int64 array of block lengths."""
    array = np.ascontiguousarray(lengths, dtype=np.int64).reshape(-1)
    out = np.zeros(1, dtype=np.int64)
    lib.bl_bisect_left(_addr(array), array.size, int(value), _addr(out))
    return int(out[0])


def insert_sorted(lengths, value: int) -> np.ndarray:
    """`bisect.insort` on a sorted int64 array; returns a new sorted array."""
    array = np.ascontiguousarray(lengths, dtype=np.int64).reshape(-1)
    out = np.empty(array.size + 1, dtype=np.int64)
    out[: array.size] = array
    new_size = lib.bl_insert_sorted(_addr(out), array.size, int(value), out.size)
    return out[:new_size].copy()


def remove_sorted(lengths, value: int) -> np.ndarray:
    """Remove the first entry equal to `value`, keeping the rest sorted."""
    array = np.ascontiguousarray(lengths, dtype=np.int64).reshape(-1)
    out = np.empty(max(array.size, 1), dtype=np.int64)
    out[: array.size] = array
    new_size = lib.bl_remove_sorted(_addr(out), array.size, int(value))
    return out[:new_size].copy()


def prefix_sum(counts) -> np.ndarray:
    """Exclusive prefix sum with the total appended, length ``len(counts) + 1``."""
    array = np.ascontiguousarray(counts, dtype=np.int64).reshape(-1)
    out = np.empty(array.size + 1, dtype=np.int64)
    lib.bl_prefix_sum(_addr(array), array.size, _addr(out))
    return out


def merge_intervals(starts, stops) -> tuple[np.ndarray, np.ndarray]:
    """Merge touching or overlapping half-open intervals, in place by start.

    Inputs must be sorted by start. Returns the merged starts and stops.
    """
    lo = np.ascontiguousarray(starts, dtype=np.int64).reshape(-1)
    hi = np.ascontiguousarray(stops, dtype=np.int64).reshape(-1)
    if lo.size != hi.size:
        raise ValueError("starts and stops must be the same length")
    if lo.size == 0:
        return lo.copy(), hi.copy()
    if lo.size > 1 and np.any(np.diff(lo) < 0):
        raise ValueError("intervals must be sorted by start")
    merged_lo = np.empty(lo.size, dtype=np.int64)
    merged_hi = np.empty(hi.size, dtype=np.int64)
    written = lib.bl_merge_intervals(
        _addr(lo), _addr(hi), lo.size, _addr(merged_lo), _addr(merged_hi)
    )
    return merged_lo[:written].copy(), merged_hi[:written].copy()
