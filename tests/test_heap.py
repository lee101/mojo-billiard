"""Parity tests for the free-list kernels against the `bisect` calls upstream makes.

`billiard.heap.Heap` keeps its free blocks in `self._lengths`, a list kept
sorted with `bisect.insort` and searched with `bisect.bisect_left`. Those two
calls are the whole search structure, and the tests compare the compiled
versions against them directly, plus against `Heap` itself for the parts of its
behaviour that do not need an mmap-backed arena.
"""

import bisect
import mmap
import sys

import numpy as np
import pytest

import mojo_billiard as mbl
from billiard.heap import Heap

SORTED = [1, 2, 2, 3, 5, 8, 8, 13, 21, 34, 55, 89]


@pytest.mark.parametrize("value", [-5, 0, 1, 2, 4, 8, 22, 55, 90, 10**6])
def test_bisect_left_matches_the_bisect_module(value):
    assert mbl.bisect_left(SORTED, value) == bisect.bisect_left(SORTED, value)


@pytest.mark.parametrize("value", SORTED + [0, 4, 22, 90])
def test_insert_sorted_matches_bisect_insort(value):
    expected = list(SORTED)
    bisect.insort(expected, value)
    np.testing.assert_array_equal(mbl.insert_sorted(SORTED, value), np.array(expected))


def test_remove_sorted_matches_list_remove():
    for value in (2, 8, 1, 89, 4, 1000):
        expected = list(SORTED)
        if value in expected:
            expected.remove(value)
        got = mbl.remove_sorted(SORTED, value)
        np.testing.assert_array_equal(got, np.array(expected, dtype=np.int64))


def test_prefix_sum_matches_numpy():
    counts = [3, 0, 5, 7, 1]
    expected = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.testing.assert_array_equal(mbl.prefix_sum(counts), expected)
    assert mbl.prefix_sum([]).tolist() == [0]


def test_merge_intervals_merges_touching_and_overlapping_runs():
    starts = [0, 8, 20, 30, 44]
    stops = [8, 16, 25, 44, 50]

    # [0,8) and [8,16) touch; [20,25) stands alone; [30,44) and [44,50) touch.
    lo, hi = mbl.merge_intervals(starts, stops)
    assert lo.tolist() == [0, 20, 30]
    assert hi.tolist() == [16, 25, 50]


def test_merge_intervals_keeps_gaps_and_single_intervals():
    lo, hi = mbl.merge_intervals([0, 100], [10, 200])
    assert lo.tolist() == [0, 100]
    assert hi.tolist() == [10, 200]
    lo, hi = mbl.merge_intervals([], [])
    assert lo.tolist() == [] and hi.tolist() == []
    lo, hi = mbl.merge_intervals([5], [9])
    assert lo.tolist() == [5] and hi.tolist() == [9]


def test_merge_intervals_rejects_unsorted_input():
    with pytest.raises(ValueError):
        mbl.merge_intervals([10, 0], [20, 5])
    with pytest.raises(ValueError):
        mbl.merge_intervals([0, 1], [1])


def test_roundup_matches_the_heap_helper():
    for n in (0, 1, 7, 8, 9, 4095, 4096):
        assert mbl.roundup(n, Heap._alignment) == Heap._roundup(n, Heap._alignment)


def test_roundup_matches_mmap_pagesize():
    for n in (1, 100, 4095, 4097):
        assert mbl.roundup(n, mmap.PAGESIZE) == Heap._roundup(n, mmap.PAGESIZE)


def test_best_fit_picks_the_same_block_heap_would():
    """`_malloc` takes `self._lengths[bisect_left(self._lengths, size)]`."""
    heap = Heap()
    heap._lengths = [8, 16, 32, 64]
    for size in (1, 8, 9, 16, 33, 64):
        expected = heap._lengths[bisect.bisect_left(heap._lengths, size)]
        assert mbl.best_fit(heap._lengths, size) == expected


def test_best_fit_reports_no_room_when_the_list_runs_out():
    assert mbl.best_fit([4, 8], 9) == -1
    assert mbl.best_fit([], 1) == -1


def test_free_list_matches_a_model_built_from_bisect():
    """A randomised malloc/free workload against the same calls upstream makes."""
    rng = np.random.default_rng(7)
    free = mbl.FreeList()
    model = []
    live = []
    for step in range(3000):
        if not live or rng.random() < 0.55:
            size = int(rng.integers(8, 512))
            expected = model[bisect.bisect_left(model, size)] if \
                bisect.bisect_left(model, size) < len(model) else None
            got = free.find(size)
            if expected is None:
                assert got == -1
            else:
                assert got == expected
                model.remove(expected)
                live.append((got, size))
        else:
            index = int(rng.integers(0, len(live)))
            block, size = live.pop(index)
            free.add(block)
            bisect.insort(model, block)
        assert free.lengths.tolist() == model
        # Every live block must be at least as large as what it was asked for,
        # which is the invariant `_malloc` maintains.
        for block, size in live:
            assert block >= size
    assert len(free) == len(model)


def test_free_list_capacity_is_respected():
    free = mbl.FreeList(capacity=4)
    for value in (5, 3, 9, 1):
        free.add(value)
    assert free.lengths.tolist() == [1, 3, 5, 9]
    free.add(7)
    assert free.lengths.tolist() == [1, 3, 5, 9]
    free.add(0)
    assert free.lengths.tolist() == [1, 3, 5, 9]


def test_free_list_sorting_its_input():
    free = mbl.FreeList([9, 1, 5])
    assert free.lengths.tolist() == [1, 5, 9]
    assert len(free) == 3


def test_insert_sorted_into_an_empty_list():
    assert mbl.insert_sorted([], 1).tolist() == [1]
    assert mbl.insert_sorted([4], 2).tolist() == [2, 4]


def test_the_real_heap_still_works_unchanged():
    """A smoke test that the installed upstream package is what the tests assume."""
    heap = Heap()
    block = heap.malloc(100)
    assert block[2] - block[1] >= 100
    heap.free(block)
    assert not heap._allocated_blocks
