#!/usr/bin/env python3
"""
Break down per-call GPU allocation by `_compute.py` function for a slice.

test_missing_repeat_memory's second assertion bounds total transient allocation
(`_allocated_bytes(fn) < 6 * output_bytes`). The pure-CuPy rewrites are leak-free
but allocate intermediates; this script attributes the gross malloc volume of
one `array[:, slicer]` to each `_compute.py` function so we know which to slim
down (the backend resolves `cuda_compute.<name>` fresh per call, so patching the
module attributes is picked up).

Run on the GPU box:  python benchmarks/diagnose_alloc_by_function.py
"""

from __future__ import annotations

import collections
import functools

import numpy as np

try:
    import cupy as cp

    import awkward as ak
    import awkward._connect.cuda._compute as C
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"needs a GPU box with cupy + awkward: {exc!r}") from exc


_totals: collections.Counter = collections.Counter()
_calls: collections.Counter = collections.Counter()


class _Rec(cp.cuda.MemoryHook):
    name = "alloc-by-fn"

    def __init__(self):
        self.total = 0

    def malloc_postprocess(self, **kwargs):
        self.total += kwargs.get("mem_size", kwargs.get("size", 0))


def _wrap(fn, name):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        rec = _Rec()
        with rec:
            result = fn(*args, **kwargs)
            cp.cuda.Device().synchronize()
        _totals[name] += rec.total
        _calls[name] += 1
        return result

    return inner


def _install():
    for name in dir(C):
        if name.startswith("awkward_") or name == "segmented_sort":
            fn = getattr(C, name)
            if callable(fn):
                setattr(C, name, _wrap(fn, name))


def build():
    rows, cols = 100_000, 16
    data = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    array = ak.to_backend(ak.Array(data), "cuda")
    slicer = [0, None, cols - 1]
    return lambda: array[:, slicer]


def main():
    _install()
    fn = build()
    fn()  # warm up JIT/caches
    cp.cuda.Device().synchronize()
    _totals.clear()
    _calls.clear()
    fn()  # measured call
    cp.cuda.Device().synchronize()

    output_bytes = 100_000 * 3 * 8
    print(
        "per-_compute-function gross allocation during one slice "
        f"(budget = 6 x output_bytes = {6 * output_bytes:,} B):\n"
    )
    for name, nbytes in _totals.most_common():
        print(f"  {nbytes:>13,} B  x{_calls[name]}  {name}")
    print(f"\n  TOTAL {sum(_totals.values()):,} B")
    print("  (nested _compute calls, if any, are double-counted)")


if __name__ == "__main__":
    main()
