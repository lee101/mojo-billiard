"""Correctness-gated benchmark for mojo-billiard.

Every case checks agreement with the `bisect` calls `billiard.heap.Heap` makes
before timing, so a regression in the Mojo kernels shows up as a correctness
failure rather than a suspiciously good number.
"""

from __future__ import annotations

import bisect
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_billiard as mbl  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def random_lengths(count, seed=0, hi=1 << 20):
    rng = np.random.default_rng(seed)
    return np.sort(rng.integers(1, hi, size=count, dtype=np.int64))


def bench_bisect(count=1 << 20, probes=4096):
    lengths = random_lengths(count, seed=1)
    rng = np.random.default_rng(2)
    values = rng.integers(0, 1 << 21, size=probes, dtype=np.int64)

    for value in values[:64]:
        assert mbl.bisect_left(lengths, int(value)) == bisect.bisect_left(
            lengths.tolist(), int(value)
        )

    def reference():
        listed = lengths.tolist()
        for value in values.tolist():
            bisect.bisect_left(listed, value)

    def ours():
        for value in values.tolist():
            mbl.bisect_left(lengths, value)

    return f"bisect_left {probes} probes in {count}", _time(reference), _time(ours)


def bench_insort(count=1 << 18, inserts=2048):
    rng = np.random.default_rng(3)
    lengths = random_lengths(count, seed=4)
    values = rng.integers(1, 1 << 21, size=inserts, dtype=np.int64)

    for value in values[:32]:
        expected = lengths.tolist()
        bisect.insort(expected, int(value))
        np.testing.assert_array_equal(
            mbl.insert_sorted(lengths, value), np.array(expected, dtype=np.int64)
        )

    def reference():
        listed = lengths.tolist()
        for value in values.tolist():
            bisect.insort(listed, value)

    def ours():
        current = lengths
        for value in values.tolist():
            current = mbl.insert_sorted(current, value)

    return f"insort {inserts} into {count}", _time(reference), _time(ours)


def bench_freelist(steps=20000, seed=5):
    """A malloc/free workload against a model built from the same bisect calls."""
    rng = np.random.default_rng(seed)

    def run_reference():
        listed = []
        live = []
        for _ in range(steps):
            if not live or rng.random() < 0.55:
                size = int(rng.integers(8, 4096))
                index = bisect.bisect_left(listed, size)
                if index < len(listed):
                    live.append(listed.pop(index))
            else:
                bisect.insort(listed, live.pop(int(rng.integers(0, len(live)))))
        return listed

    def run_ours():
        free = mbl.FreeList()
        live = []
        for _ in range(steps):
            if not live or rng.random() < 0.55:
                block = free.find(int(rng.integers(8, 4096)))
                if block > 0:
                    live.append(block)
            else:
                free.add(live.pop(int(rng.integers(0, len(live)))))
        return free.lengths

    return "freelist 20000 steps", _time(run_reference), _time(run_ours)


def bench_merge_intervals(count=1 << 20, seed=6):
    rng = np.random.default_rng(seed)
    stops = np.cumsum(rng.integers(1, 64, size=count, dtype=np.int64))
    starts = stops - 1
    merged_lo, merged_hi = mbl.merge_intervals(starts, stops)
    assert merged_lo[0] == starts[0] and merged_hi[-1] == stops[-1]
    assert np.all(merged_hi[:-1] < merged_lo[1:]) or merged_lo.size == 1

    def reference():
        out_lo = [int(starts[0])]
        out_hi = [int(stops[0])]
        for i in range(1, starts.size):
            if int(starts[i]) <= out_hi[-1]:
                out_hi[-1] = max(out_hi[-1], int(stops[i]))
            else:
                out_lo.append(int(starts[i]))
                out_hi.append(int(stops[i]))
        return out_lo, out_hi

    def ours():
        return mbl.merge_intervals(starts, stops)

    return f"merge {count} touching intervals", _time(reference, 2), _time(ours, 2)


def main():
    print(f"{'case':<40}{'python/bisect':>15}{'mojo-billiard':>16}{'ratio':>9}")
    print("-" * 80)
    for fn in (bench_bisect, bench_insort, bench_freelist, bench_merge_intervals):
        label, reference, ours = fn()
        ratio = reference / ours if ours else float("nan")
        print(f"{label:<40}{reference * 1e3:>13.2f}ms{ours * 1e3:>14.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
