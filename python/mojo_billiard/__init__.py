"""mojo-billiard: the block-allocator bookkeeping of `billiard` in Mojo.

`billiard` is the process pool and IPC library Celery is built on: worker
processes, pipes, shared memory, semaphores, and the block allocator that
`sharedctypes.RawValue` uses. Almost all of it is process management and IO,
which a compiled library cannot help with. The one part that is arithmetic over
arrays of integers is the allocator's free list: the sorted list of free-block
lengths that `billiard.heap.Heap` keeps with `bisect.insort` and searches with
`bisect.bisect_left`, and the interval coalescing that goes with it. That is
what this port implements.

The Python layer keeps the blocks in sorted NumPy arrays rather than in the two
dictionaries upstream uses (`_start_to_block`, `_stop_to_block`); a dictionary
is a hash table and does not belong in a kernel. The search, the insert, the
remove and the coalescing sweep are the compiled parts, and the parity tests
compare them against the exact `bisect` calls upstream makes.

The process pool, the pipes, the connection framing, the semaphores, the shared
ctypes wrappers, the heaps that back `RawArray`/`RawValue`, and the CLI are all
left to the real `billiard`, which is installed alongside this package.
"""

from __future__ import annotations

import numpy as np

from . import _lib

__all__ = [
    "FreeList",
    "best_fit",
    "bisect_left",
    "insert_sorted",
    "merge_intervals",
    "prefix_sum",
    "remove_sorted",
    "roundup",
]

bisect_left = _lib.bisect_left
insert_sorted = _lib.insert_sorted
remove_sorted = _lib.remove_sorted
prefix_sum = _lib.prefix_sum
merge_intervals = _lib.merge_intervals


def roundup(n: int, alignment: int) -> int:
    """`Heap._roundup`: round `n` up to a multiple of a power-of-two `alignment`."""
    mask = alignment - 1
    return (n + mask) & ~mask


def best_fit(lengths, size: int) -> int:
    """The smallest free block at least `size` long, or ``-1`` if there is none.

    This is `Heap._malloc`'s search: ``bisect_left`` on the sorted lengths, and
    a new arena when the index runs off the end.
    """
    array = np.ascontiguousarray(lengths, dtype=np.int64).reshape(-1)
    index = bisect_left(array, size)
    if index == array.size:
        return -1
    return int(array[index])


class FreeList:
    """A best-fit free list of block lengths, as `billiard.heap.Heap` keeps one.

    Blocks are tracked as lengths only, which is all the search needs; the
    coalescing that upstream does with its two block dictionaries is exposed
    separately as :func:`merge_intervals`.
    """

    __slots__ = ("_lengths", "_capacity")

    def __init__(self, lengths=(), capacity: int = 1 << 16):
        self._capacity = int(capacity)
        ordered = np.sort(np.ascontiguousarray(list(lengths), dtype=np.int64))
        self._lengths = ordered[: self._capacity].copy()

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths.copy()

    def __len__(self) -> int:
        return int(self._lengths.size)

    def find(self, size: int) -> int:
        """Take the best-fit block for `size`, or return ``-1`` and change nothing."""
        index = bisect_left(self._lengths, size)
        if index == self._lengths.size:
            return -1
        taken = int(self._lengths[index])
        self._lengths = remove_sorted(self._lengths, taken)
        return taken

    def add(self, length: int) -> np.ndarray:
        """Return a free block of `length` to the list."""
        if length <= 0:
            return self._lengths
        if self._lengths.size >= self._capacity:
            return self._lengths
        self._lengths = insert_sorted(self._lengths, int(length))
        return self._lengths
