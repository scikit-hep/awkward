# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
"""
awkward_ListArray_combinations — faithful port of the PR reference implementation.

Only n=2 without replacement is supported via the cuda.compute path.
All other (n, replacement) combinations raise NotImplementedError.

Pipeline (all cuda.compute, no RawKernel):
  1. binary_transform(stops, starts, counts, count_pairs_from_stops_starts)
  3. exclusive_scan(counts, out_offsets[:length], PLUS, 0, length)
     out_offsets[0] = 0 (pre-set), out_offsets[length] = total (post-set)
  4. searchsorted(out_offsets, arange(total)) → seg — parent list per output element
     unary_transform(ZipIterator(g, b, nn, out_base)[seg], ZipIterator(out0,out1), _unrank_k2)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import cupy as cp
import numpy as np

import cuda.compute as cc
from cuda.compute import CountingIterator, OpKind, TransformIterator, ZipIterator


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _count_pairs_from_stops_starts(pair):
    # pair = (stops_i, starts_i); compute C(stops-starts, 2) in one step
    n = pair[0] - pair[1]
    return (n * (n - 1) // 2) if n >= 2 else np.int64(0)


@lru_cache(maxsize=32)
def _make_unrank_k2_cached(length):
    """
    Build and cache the unrank op keyed on `length` (number of lists).

    _len is a scalar constant baked into the closure at creation time.
    cuda.compute caches by (bytecode + closure cell values).  The three
    array cells (_off, _starts, _stops) are pre-allocated here with fixed
    Python identity; their *contents* are updated each call via cp.copyto,
    so the closure cell values (the array objects) never change — the
    cuda.compute cache always hits after the first compile for each length.
    """
    _len    = np.int64(length)
    # Placeholder arrays — fixed Python objects, contents swapped each call
    _off    = cp.empty(length + 1, dtype=np.int64)
    _starts = cp.empty(length,     dtype=np.int64)
    _stops  = cp.empty(length,     dtype=np.int64)

    def fn(g):
        # Binary search _off[1..length] for parent list of output index g
        lo = np.int64(0)
        hi = _len - np.int64(1)
        while lo < hi:
            mid = (lo + hi) >> np.int64(1)
            if _off[mid + np.int64(1)] <= g:
                lo = mid + np.int64(1)
            else:
                hi = mid
        b            = _starts[lo]
        nn           = _stops[lo] - _starts[lo]
        out_base_val = _off[lo]

        r    = g - out_base_val
        bf   = np.float64(2 * nn - 1)
        disc = bf * bf - np.float64(8) * np.float64(r)
        ii   = np.int64((bf - disc**0.5) * 0.5)

        while ii > np.int64(0):
            Si = ii * (np.int64(2) * nn - ii - np.int64(1)) // np.int64(2)
            if Si <= r:
                break
            ii -= np.int64(1)

        while (ii + np.int64(1)) < nn:
            i2  = ii + np.int64(1)
            Si2 = i2 * (np.int64(2) * nn - i2 - np.int64(1)) // np.int64(2)
            if Si2 > r:
                break
            ii = i2

        Si = ii * (np.int64(2) * nn - ii - np.int64(1)) // np.int64(2)
        t  = r - Si
        return (b + ii, b + ii + np.int64(1) + t)

    return fn, _off, _starts, _stops


# ─────────────────────────────────────────────────────────────────────────────
# combinations_length  (Pass 1 only — returns offsets + total)
# ─────────────────────────────────────────────────────────────────────────────

def combinations_length(starts64, stops64, length, _off):
    """
    Compute out_offsets and total for n=2 no-replacement.
    Writes directly into _off (the cached buffer) — no extra allocation.

    Parameters
    ----------
    starts64, stops64 : already cp.int64 arrays
    length            : int
    _off              : pre-allocated cp.int64 array of shape (length+1,)
                        from _make_unrank_k2_cached — written in-place

    Returns
    -------
    total : int
    """
    # Lazy iterator: counts[i] = C(stops[i]-starts[i], 2) — no intermediate array
    counts_iter = TransformIterator(ZipIterator(stops64, starts64),
                                    _count_pairs_from_stops_starts)

    # exclusive_scan always writes identity (0) at index 0 — no need to pre-set
    init = np.array([0], dtype=np.int64)
    cc.exclusive_scan(counts_iter, _off[:length], OpKind.PLUS, init, length)

    if length > 0:
        last_len   = int(stops64[length - 1].item()) - int(starts64[length - 1].item())
        last_count = (last_len * (last_len - 1) // 2) if last_len >= 2 else 0
        total      = int(_off[length - 1].item()) + last_count
    else:
        total = 0
    _off[length] = np.int64(total)

    return total


# ─────────────────────────────────────────────────────────────────────────────
# combinations  (Pass 2 — writes into out0, out1)
# ─────────────────────────────────────────────────────────────────────────────

def combinations(starts64, stops64, total, length, fn, _off, _starts, _stops):
    """
    Write gather indices for all pairs into out0, out1.

    _off is already filled by combinations_length (written in-place).
    _starts/_stops are updated once via copyto — same objects as the
    closure captures, so cuda.compute cache always hits.

    Returns
    -------
    out0, out1 : CuPy int64 arrays of shape (total,)
    """
    out0 = cp.empty(total, dtype=np.int64)
    out1 = cp.empty(total, dtype=np.int64)

    if total == 0:
        return out0, out1

    # _off already has correct data (written by combinations_length)
    # Only update _starts/_stops — two copies of length*8 bytes each
    cp.copyto(_starts, starts64)
    cp.copyto(_stops,  stops64)

    cc.unary_transform(
        CountingIterator(np.int64(0)),
        ZipIterator(out0, out1),
        fn,
        total,
    )

    return out0, out1


# ─────────────────────────────────────────────────────────────────────────────
# Public convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def awkward_ListArray_combinations(
    starts:      cp.ndarray,
    stops:       cp.ndarray,
    n:           int,
    replacement: bool = False,
) -> Tuple[List[cp.ndarray], cp.ndarray]:
    """
    Compute all n-combinations for each list described by starts/stops.

    Only n=2, replacement=False is supported via the cuda.compute path.
    """
    if n != 2 or replacement:
        raise NotImplementedError(
            f"cuda.compute combinations only supports n=2 without replacement, "
            f"got n={n}, replacement={replacement}"
        )

    length = len(starts)

    if length == 0:
        return [cp.zeros(0, dtype=np.int64), cp.zeros(0, dtype=np.int64)], \
               cp.zeros(2, dtype=np.int64)

    starts64 = cp.asarray(starts, dtype=np.int64)
    stops64  = cp.asarray(stops,  dtype=np.int64)

    fn, _off, _starts, _stops = _make_unrank_k2_cached(length)
    total = combinations_length(starts64, stops64, length, _off)

    if total == 0:
        return [cp.zeros(0, dtype=np.int64), cp.zeros(0, dtype=np.int64)], \
               cp.zeros(2, dtype=np.int64)

    out0, out1 = combinations(starts64, stops64, total, length, fn, _off, _starts, _stops)

    return [out0, out1], cp.full(2, total, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test + benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    print("=" * 60)
    print("awkward_ListArray_combinations (n=2, no replacement)")
    print("=" * 60)

    # Test 1: lists [[0,1,2],[3,4]]
    starts = cp.array([0, 3], dtype=np.int64)
    stops  = cp.array([3, 5], dtype=np.int64)
    tocarry, toindex = awkward_ListArray_combinations(starts, stops, n=2)
    c0, c1 = tocarry[0].get(), tocarry[1].get()
    print(f"\nTest 1 — [[0,1,2],[3,4]], total={len(c0)}")
    for i in range(len(c0)):
        print(f"  combo[{i}] = ({c0[i]}, {c1[i]})")
    assert list(c0) == [0, 0, 1, 3], list(c0)
    assert list(c1) == [1, 2, 2, 4], list(c1)
    print("  PASS")

    # Test 2: single list [0,1,2,3]
    starts = cp.array([0], dtype=np.int64)
    stops  = cp.array([4], dtype=np.int64)
    tocarry, _ = awkward_ListArray_combinations(starts, stops, n=2)
    combos = list(zip(tocarry[0].get(), tocarry[1].get()))
    print(f"\nTest 2 — [[0,1,2,3]]: {combos}")
    assert combos == [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], combos
    print("  PASS")

    # Test 3: empty first list
    tocarry, _ = awkward_ListArray_combinations(
        cp.array([0, 3], dtype=np.int64), cp.array([0, 5], dtype=np.int64), n=2)
    c0, c1 = tocarry[0].get(), tocarry[1].get()
    assert list(c0) == [3] and list(c1) == [4], (list(c0), list(c1))
    print("\nTest 3 — empty first list: PASS")

    # Test 4: all empty lists
    tocarry, _ = awkward_ListArray_combinations(
        cp.array([0, 3], dtype=np.int64), cp.array([0, 3], dtype=np.int64), n=2)
    assert len(tocarry[0]) == 0
    print("Test 4 — all empty: PASS")

    # Test 5: singleton lists (no pairs possible)
    tocarry, _ = awkward_ListArray_combinations(
        cp.array([0, 1], dtype=np.int64), cp.array([1, 2], dtype=np.int64), n=2)
    assert len(tocarry[0]) == 0
    print("Test 5 — all singletons: PASS")

    # Test 6: large smoke test
    rng   = np.random.default_rng(42)
    sizes = rng.integers(0, 20, size=100_000).astype(np.int64)
    hs    = np.concatenate([[0], np.cumsum(sizes[:-1])])
    tocarry, toindex = awkward_ListArray_combinations(
        cp.asarray(hs), cp.asarray(hs + sizes), n=2)
    total = int(toindex[0].get())
    # Verify total matches CPU reference
    expected = int(np.sum(sizes * (sizes - 1) // 2))
    assert total == expected, f"{total} != {expected}"
    print(f"\nTest 6 — 100k lists, n=2: total={total:,}  PASS")

    # Test 7: unsupported n raises
    try:
        awkward_ListArray_combinations(
            cp.array([0], dtype=np.int64), cp.array([4], dtype=np.int64), n=3)
        assert False, "should have raised"
    except NotImplementedError:
        pass
    print("Test 7 — n=3 raises NotImplementedError: PASS")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)

    # Benchmark
    import time
    print("\nBenchmark: 1M lists, uniform size 10, n=2, no replacement")
    sizes2 = np.full(1_000_000, 10, dtype=np.int64)
    hs     = np.concatenate([[0], np.cumsum(sizes2[:-1])])
    ds, de = cp.asarray(hs), cp.asarray(hs + sizes2)

    for _ in range(5):
        awkward_ListArray_combinations(ds, de, n=2)
    cp.cuda.Stream.null.synchronize()

    TIMED_ITERS = 20
    times = []
    for _ in range(TIMED_ITERS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        awkward_ListArray_combinations(ds, de, n=2)
        cp.cuda.Stream.null.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    trim    = max(len(times) // 10, 1)
    trimmed = times[trim:-trim]
    print(f"  min:              {times[0]:.2f} ms")
    print(f"  median:           {times[len(times)//2]:.2f} ms")
    print(f"  trimmed mean 80%: {sum(trimmed)/len(trimmed):.2f} ms")
    print(f"  max:              {times[-1]:.2f} ms")


if __name__ == "__main__":
    _run_tests()
