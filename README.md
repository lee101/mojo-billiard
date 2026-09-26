# mojo-billiard

`mojo-billiard` is a Mojo port of the block-allocator bookkeeping inside
[`billiard`](https://pypi.org/project/billiard/), the process pool and IPC
library Celery is built on. The Python package is named `mojo_billiard`, so it
installs alongside the real one and the tests compare against the exact `bisect`
calls upstream makes.

```python
import mojo_billiard as mbl

free = mbl.FreeList([8192, 16384, 16384])
free.find(10000)            # 16384: the best fit, as Heap._malloc does it
free.add(4096)              # back into the sorted length list
mbl.merge_intervals([0, 8], [8, 20])     # (array([0]), array([20])): coalesced
mbl.prefix_sum([3, 0, 5])   # run offsets of a bucketed free list
mbl.roundup(4095, 4096)     # 4096, exactly Heap._roundup
```

## What is ported, and why

`billiard` is process management: workers, pipes, shared memory, semaphores,
pickle, forking. None of that is arithmetic a compiled library can help with,
and saying so is more useful than inventing a kernel for it.

There is one part that *is* array work. `billiard.heap.Heap` hands out blocks
from mmap-backed arenas and keeps its free blocks in `self._lengths`, a list
kept sorted with `bisect.insort` so that an allocation can find the best fit
with a single `bisect.bisect_left`. `billiard.pool` relies on the same idea
with power-of-two size buckets and one run per bucket. Both are integer arrays
with sorted-search and shift-heavy insertion, and that is what is compiled here:

| area | implemented API | kernel |
| --- | --- | --- |
| Best-fit search | `bisect_left`, `best_fit`, `FreeList.find` | `bl_bisect_left` |
| Free-list bookkeeping | `insert_sorted`, `remove_sorted`, `FreeList.add` | `bl_insert_sorted`, `bl_remove_sorted` |
| Bucketed runs | `prefix_sum` | `bl_prefix_sum` |
| Block coalescing | `merge_intervals` | `bl_merge_intervals` |

Upstream keeps its blocks in two dictionaries, `_start_to_block` and
`_stop_to_block`, so that a free can look up its left and right neighbours and
absorb them. A dictionary is a hash table and does not belong in a kernel, so
this port keeps the blocks in sorted arrays and moves the *sweep* -- the part
that decides which neighbours touch and which runs merge -- into Mojo.

## Not implemented

Everything else in `billiard`: the process pool and its worker lifecycle, the
pipes and `Connection` framing, the `SimpleQueue` ring buffer, the semaphores
and condition variables, the reduction and shared-ctypes machinery, the
spawn/forkserver/exec start methods, the `mmap`-backed `Arena` itself, the
resource tracker, and the `billiard.connection`/`pool` command-line tooling.
`Arena` is left to the real package because it is an mmap wrapper, not an array
kernel; `bl_bisect_left` returns the "index ran off the end" sentinel that
`Heap._malloc` uses to decide to grow a new arena, and the tests check that
sentinel against the real `Heap`.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-billiard.so`. Set `PYTHONPATH=python`
outside a Pixi task. With the shared toolchain:

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

`pytest.ini` disables a broken `zarr` plugin in the shared test venv (it needs a
NumPy 2 symbol the venv's NumPy 1.26 does not have). The plugin is unrelated to
these tests and aborts collection before any test runs.

## Performance

Best-of-five wall clock, same process, against the `bisect` module and the Python
list operations upstream uses. Every case checks agreement before timing.

| case | python / bisect | mojo-billiard | result |
| --- | ---: | ---: | ---: |
| merge 1048576 touching intervals | 1844.97 ms | 23.13 ms | 79.8x faster |
| bisect_left 4096 probes in 1048576 | 99.27 ms | 47.39 ms | 2.09x faster |
| freelist 20000 malloc/free steps | 133.16 ms | 422.24 ms | 0.32x, **slower** |
| insort 2048 into 262144 | 87.22 ms | 1164.98 ms | 0.07x, **slower** |

Two real wins and two real losses, and the losses are worth explaining rather
than hiding. `bl_merge_intervals` is 80x because the Python reference builds two
Python lists element by element over a million intervals while the kernel makes
one pass over two contiguous int64 arrays. `bl_bisect_left` is 2x on a million
element array because `bisect.bisect_left` boxes every comparison against a
Python int.

The two losses are the shim's fault, not the kernel's: `insert_sorted` and
`remove_sorted` in `_lib.py` return a **new** array, so every insert copies the
whole list, and that copy dominates the `O(n)` shift the kernel is there to
perform. The kernel already supports an in-place update with a capacity bound
(`bl_insert_sorted` stops rather than writing past the buffer and returns the
unchanged length), and `FreeList` deliberately does not use it yet, so this is a
known and localised gap: the next step is a `FreeList` that owns a
capacity-sized buffer and calls the kernel in place. Until then, the honest
numbers are the ones in the table.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit; `build/build.sh`
compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-billiard.so`. The `python/mojo_billiard` layer owns every array,
normalises to contiguous `int64`, and refuses unsorted input to
`merge_intervals` rather than producing a plausible wrong merge. Addresses cross
the C ABI as 64-bit integers and are rebuilt as
`Pointer[Int64, AnyOrigin[mut=True]]`.

`bl_insert_sorted` and `bl_remove_sorted` return the new length, so the caller
always knows how much of the buffer is live; both declare `-> Int abi("C")`,
because a function that both returns a value and falls off the end would
otherwise infer `None` and fail to compile.

The tests are parity tests against the mechanism upstream uses: every sorted
search is compared with `bisect.bisect_left` on the same list, every insert with
`bisect.insort`, every remove with `list.remove`, `roundup` with
`Heap._roundup`, and a randomised 3000-step malloc/free workload is run against
a model built from those same `bisect` calls with the state compared after
every single step. All of that arithmetic is integer, so the comparisons are
exact -- `rtol=0, atol=0` -- which is the right tolerance here and nowhere else
in this repository.

## License

MIT
