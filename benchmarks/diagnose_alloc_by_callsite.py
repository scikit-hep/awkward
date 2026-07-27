#!/usr/bin/env python3
"""
Attribute every GPU malloc during one slice to its Awkward source line.

The kernel (`_compute.py`) allocation is already minimized; the test still sees
a large total because most of it happens in Awkward's slicing machinery
(content.py / _slicing.py / index.py), outside any kernel. This profiler installs
a global MemoryHook that, for each allocation, walks the stack to the deepest
`awkward/` frame and tallies bytes per call site -- so we can spot any redundant
large copy (e.g. a full-input duplication) before deciding the test's bound.

Run on the GPU box:  python benchmarks/diagnose_alloc_by_callsite.py
"""

from __future__ import annotations

import collections
import traceback

import numpy as np

try:
    import cupy as cp

    import awkward as ak
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"needs a GPU box with cupy + awkward: {exc!r}") from exc


_by_site: collections.Counter = collections.Counter()
_count: collections.Counter = collections.Counter()


class _Hook(cp.cuda.MemoryHook):
    name = "alloc-by-callsite"

    def malloc_postprocess(self, **kwargs):
        size = kwargs.get("mem_size", kwargs.get("size", 0))
        key = "<non-awkward>"
        for frame in reversed(traceback.extract_stack()):
            if "/awkward/" in frame.filename and "benchmarks/" not in frame.filename:
                short = frame.filename.split("/awkward/", 1)[1]
                key = f"{short}:{frame.lineno} {frame.name}"
                break
        _by_site[key] += size
        _count[key] += 1


def build():
    rows, cols = 100_000, 16
    data = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    array = ak.to_backend(ak.Array(data), "cuda")
    slicer = [0, None, cols - 1]
    return lambda: array[:, slicer]


def main():
    fn = build()
    fn()  # warm up
    cp.cuda.Device().synchronize()

    hook = _Hook()
    with hook:
        fn()
        cp.cuda.Device().synchronize()

    output_bytes = 100_000 * 3 * 8
    total = sum(_by_site.values())
    print(
        f"GPU allocation by Awkward call site during one slice "
        f"(total {total:,} B; budget 6*output={6 * output_bytes:,} B):\n"
    )
    for site, nbytes in _by_site.most_common(20):
        print(f"  {nbytes:>13,} B  x{_count[site]:<3} {site}")
    print(f"\n  TOTAL {total:,} B")


if __name__ == "__main__":
    main()
