"""Compiled kernels for the block-allocator bookkeeping in `billiard.heap`.

`billiard.heap.Heap` hands out blocks from mmap-backed arenas and keeps its free
blocks in a list of lengths that is kept sorted with `bisect.insort` so that an
allocation can find the best fit with `bisect.bisect_left`. That sorted-length
bookkeeping and the interval coalescing that goes with it are the numeric core
of the allocator, and they are what the kernels here do.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.
"""

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]


def lp(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


@export("bl_bisect_left")
def bl_bisect_left(
    lengths_addr: Int, count: Int, value: Int, out_addr: Int
) abi("C"):
    """`bisect.bisect_left(lengths, value)`: the first index whose entry is >= value.

    Returns ``count`` when every entry is smaller, which is the "allocate a new
    arena" signal in `Heap._malloc`. An out-of-range read would be a segmentation
    fault, so the sentinel is written rather than read.
    """
    var lengths = lp(lengths_addr)
    var out = lp(out_addr)
    var lo = 0
    var hi = count
    while lo < hi:
        var mid = (lo + hi) // 2
        if lengths[unsafe_offset=mid] < Int64(value):
            lo = mid + 1
        else:
            hi = mid
    out[unsafe_offset=0] = Int64(lo)


@export("bl_insert_sorted")
def bl_insert_sorted(
    lengths_addr: Int, count: Int, value: Int, capacity: Int
) abi("C") -> Int:
    """`bisect.insort(lengths, value)`, in place; returns the new length.

    A capacity overrun is a caller error, and the kernel stops rather than
    writing past the buffer, returning the unchanged count so the caller can
    detect it.
    """
    var lengths = lp(lengths_addr)
    if count >= capacity:
        return count
    var lo = 0
    var hi = count
    while lo < hi:
        var mid = (lo + hi) // 2
        if lengths[unsafe_offset=mid] < Int64(value):
            lo = mid + 1
        else:
            hi = mid
    var i = count
    while i > lo:
        lengths[unsafe_offset=i] = lengths[unsafe_offset=i - 1]
        i -= 1
    lengths[unsafe_offset=lo] = Int64(value)
    return count + 1


@export("bl_remove_sorted")
def bl_remove_sorted(
    lengths_addr: Int, count: Int, value: Int
) abi("C") -> Int:
    """Remove the first entry equal to `value`; returns the new length.

    `Heap._absorb` does `self._lengths.remove(length)` and only removes the
    bucket from the list when its block sequence empties, so this mirrors
    "remove one occurrence, keep the rest in order".
    """
    var lengths = lp(lengths_addr)
    var lo = 0
    var hi = count
    while lo < hi:
        var mid = (lo + hi) // 2
        if lengths[unsafe_offset=mid] < Int64(value):
            lo = mid + 1
        else:
            hi = mid
    if lo >= count or lengths[unsafe_offset=lo] != Int64(value):
        return count
    var i = lo
    while i < count - 1:
        lengths[unsafe_offset=i] = lengths[unsafe_offset=i + 1]
        i += 1
    return count - 1


@export("bl_prefix_sum")
def bl_prefix_sum(
    counts_addr: Int, count: Int, out_addr: Int
) abi("C"):
    """Exclusive prefix sum, the run offsets of a bucketed free list.

    `Pool` sizes its per-worker requests in powers of two and keeps one free
    block per size bucket; the offset of each bucket's run is this prefix sum.
    Output has `count + 1` entries and the last one is the total.
    """
    var counts = lp(counts_addr)
    var out = lp(out_addr)
    out[unsafe_offset=0] = 0
    var acc = Int64(0)
    for i in range(count):
        acc += counts[unsafe_offset=i]
        out[unsafe_offset=i + 1] = acc


@export("bl_merge_intervals")
def bl_merge_intervals(
    starts_addr: Int, stops_addr: Int, count: Int, merged_starts_addr: Int,
    merged_stops_addr: Int
) abi("C") -> Int:
    """Merge touching or overlapping ``[start, stop)`` intervals, in place.

    Inputs must be sorted by start and non-overlapping only up to touching: two
    blocks are neighbours, and coalescable, when one stops exactly where the
    next begins. This is the interval merge `Heap._free` performs by looking a
    block up by its start and by its stop in two dictionaries; a dictionary is a
    hash table and does not belong in a kernel, so the Python layer keeps the
    blocks in sorted arrays and this kernel does the sweep.

    Returns the number of merged intervals; the results are written from index 0
    of the output buffers, which must hold `count` entries each.
    """
    var starts = lp(starts_addr)
    var stops = lp(stops_addr)
    var merged_starts = lp(merged_starts_addr)
    var merged_stops = lp(merged_stops_addr)
    if count == 0:
        return 0
    var written: Int = 1
    merged_starts[unsafe_offset=0] = starts[unsafe_offset=0]
    merged_stops[unsafe_offset=0] = stops[unsafe_offset=0]
    for i in range(1, count):
        var start = starts[unsafe_offset=i]
        var stop = stops[unsafe_offset=i]
        if start <= merged_stops[unsafe_offset=written - 1]:
            if stop > merged_stops[unsafe_offset=written - 1]:
                merged_stops[unsafe_offset=written - 1] = stop
        else:
            merged_starts[unsafe_offset=written] = start
            merged_stops[unsafe_offset=written] = stop
            written += 1
    return written
