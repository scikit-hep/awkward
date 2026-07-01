#!/usr/bin/env python3
"""
Identify what retains the per-call GPU buffers in test_missing_repeat_memory.

The cuda.compute closure-capture leak is fixed, yet the leak test still shows a
fixed ~0.8 MB-per-call growth that survives free_all_blocks. This script finds
the actual holder the same way the original PR #4056 reducer leak was found:
run the operation in a loop, then walk gc referrers of the surviving CuPy
buffers and print the chain of holder types.

Run on the GPU box:  python benchmarks/diagnose_retained_buffers.py
"""

from __future__ import annotations

import collections
import gc

import numpy as np

try:
    import cupy as cp

    import awkward as ak
    import awkward._connect.cuda as ak_cuda
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"needs a GPU box with cupy + awkward: {exc!r}") from exc


def build():
    rows, cols = 100_000, 16
    data = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    array = ak.to_backend(ak.Array(data), "cuda")
    slicer = [0, None, cols - 1]
    return array, (lambda: array[:, slicer])


def drain_awkward_contexts():
    """Flush Awkward's pending CUDA-kernel Invocation list, if present."""
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    if stream.ptr in ak_cuda.cuda_streamptr_to_contexts:
        ak_cuda.synchronize_cuda(stream)


def used():
    return cp.get_default_memory_pool().used_bytes()


def report(tag, do_drain):
    array, fn = build()
    pool = cp.get_default_memory_pool()

    fn()
    cp.cuda.Device().synchronize()
    if do_drain:
        drain_awkward_contexts()
    gc.collect()
    pool.free_all_blocks()
    baseline = used()

    for _ in range(5):
        fn()
    cp.cuda.Device().synchronize()
    if do_drain:
        drain_awkward_contexts()
    gc.collect()
    pool.free_all_blocks()
    after = used()

    print(f"\n=== {tag} (drain_awkward_contexts={do_drain}) ===")
    print(f"baseline={baseline:,}  after={after:,}  delta={after - baseline:,}")

    # Surviving CuPy buffers grouped by size.
    arrays = [o for o in gc.get_objects() if isinstance(o, cp.ndarray)]
    by_size = collections.Counter(a.nbytes for a in arrays)
    print("live cupy arrays by nbytes (largest first):")
    for nbytes, count in sorted(by_size.items(), reverse=True)[:12]:
        print(f"  {nbytes:>12,} x{count}")

    # Referrer chains for the ~0.8 MB buffers (one input row's worth: 100k int64).
    suspects = [a for a in arrays if 700_000 <= a.nbytes <= 900_000]
    print(f"\n{len(suspects)} buffer(s) in 0.7-0.9 MB; referrer types (2 levels):")
    for a in suspects[:8]:
        level1 = gc.get_referrers(a)
        for r1 in level1:
            t1 = f"{type(r1).__module__}.{type(r1).__name__}"
            roots = []
            for r2 in gc.get_referrers(r1):
                roots.append(f"{type(r2).__module__}.{type(r2).__name__}")
            print(f"  {a.nbytes:,}: <- {t1}  <- {roots[:6]}")
            break

    # Is the Invocation list holding anything?
    ctxs = ak_cuda.cuda_streamptr_to_contexts
    pend = {ptr: len(v[1]) for ptr, v in ctxs.items()}
    print(f"pending Invocations per stream: {pend}")
    del array, fn


def main():
    report("no drain", do_drain=False)
    report("with drain", do_drain=True)


if __name__ == "__main__":
    main()
