# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
"""
awkward_ListArray_combinations — optimised Python / cuda.compute implementation.

Performance design:
  Pass A  – vectorised _binom_vec (CuPy ufunc, no Python loop over elements)
            + cuda.compute.inclusive_scan  (CUB DeviceScan under the hood)
  Pass B  – fused into a single cp.searchsorted + one CuPy subtract (no extra kernel)
  Pass C  – ONE RawKernel launch per carry component:
              • no device→host sync inside the loop
              • no per-slot cp.max().get() / cp.any().get() stalls
              • unrank runs entirely on-device, one thread per output element
              • carries are written directly into pre-allocated output arrays

The only host↔device transfer is the single scalar `total = d_offsets[-1].get()`
needed to size the output buffers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import cupy as cp
import numpy as np

import cuda.compute
from cuda.compute import OpKind, inclusive_scan

# ─────────────────────────────────────────────────────────────────────────────
# Vectorised binomial  C(d_m, k)  — stays on device, O(k) element-wise ops
# ─────────────────────────────────────────────────────────────────────────────

def _binom_vec(d_m: cp.ndarray, k: int) -> cp.ndarray:
    """C(d_m[i], k) for fixed integer k, fully vectorised on device."""
    if k == 0:
        return cp.ones(d_m.shape, dtype=np.int64)
    result = cp.ones(d_m.shape, dtype=np.int64)
    valid  = d_m >= k
    for j in range(k):
        result = cp.where(valid, result * (d_m - j) // (j + 1), cp.int64(0))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pass A — per-list counts + CUB inclusive scan
# ─────────────────────────────────────────────────────────────────────────────

def _pass_a_counts(
    starts:      cp.ndarray,
    stops:       cp.ndarray,
    n:           int,
    replacement: bool,
) -> cp.ndarray:
    """
    Returns int64 array of shape (length+1,): prefix-sum offsets
    where offsets[0]=0 and offsets[i+1]-offsets[i] = combo_count(list_i).
    """
    length  = int(starts.shape[0])
    d_m     = (stops - starts).astype(np.int64)

    if n == 0:
        d_counts = cp.ones(length, dtype=np.int64)
    elif not replacement:
        d_counts = cp.maximum(d_m, cp.int64(0)) if n == 1 else _binom_vec(d_m, n)
    else:
        d_top    = d_m + (n - 1)
        d_counts = cp.where(d_m > 0, _binom_vec(d_top, n), cp.int64(0))

    d_offsets    = cp.empty(length + 1, dtype=np.int64)
    d_offsets[0] = 0
    inclusive_scan(
        d_counts,
        d_offsets[1:],
        OpKind.PLUS,
        np.array([0], dtype=np.int64),
        length,
    )
    return d_offsets


# ─────────────────────────────────────────────────────────────────────────────
# Pass B — parent + local rank, fully vectorised, zero extra kernel launches
# ─────────────────────────────────────────────────────────────────────────────

def _pass_b(d_offsets: cp.ndarray, total: int) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Returns (d_parents, d_local) both int64 of shape (total,).
    Two device ops, no Python-level sync.
    """
    if total == 0:
        empty = cp.zeros(0, dtype=np.int64)
        return empty, empty

    d_arange  = cp.arange(total, dtype=np.int64)
    d_parents = cp.searchsorted(d_offsets[1:], d_arange, side="right").astype(np.int64)
    d_local   = d_arange - d_offsets[d_parents]
    return d_parents, d_local


# ─────────────────────────────────────────────────────────────────────────────
# Pass C — RawKernel unranking: one launch, all slots, zero Python-level sync
# ─────────────────────────────────────────────────────────────────────────────

_UNRANK_KERNEL_SRC = r"""
extern "C" __global__
void unrank_combinations(
    const long long* __restrict__ start_vals,  // starts[parent[tid]], shape (total,)
    const long long* __restrict__ m_vals,      // list-length per output, shape (total,)
    const long long* __restrict__ k_vals,      // local rank per output, shape (total,)
    long long* __restrict__ out,               // (n, total) row-major output
    const long long total,
    const int n,
    const int replacement)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const long long m     = m_vals[tid];
    const long long start = start_vals[tid];
    long long k           = k_vals[tid];

    // Inline binom(a, b) — multiplicative, integer-safe for small b
    auto binom = [](long long a, long long b) -> long long {
        if (b < 0 || b > a) return 0LL;
        if (b == 0 || b == a) return 1LL;
        if (b > a - b) b = a - b;
        long long r = 1LL;
        for (long long i = 0; i < b; ++i) r = r * (a - i) / (i + 1);
        return r;
    };

    if (!replacement) {
        long long v_min = 0;
        for (int r = 0; r < n; ++r) {
            const long long remaining = n - r - 1;
            long long v = v_min;
            while (v < m) {
                const long long c = binom(m - v - 1, remaining);
                if (c <= k) { k -= c; ++v; } else break;
            }
            out[(long long)r * total + tid] = start + v;
            v_min = v + 1;
        }
    } else {
        long long v_min = 0;
        for (int r = 0; r < n; ++r) {
            const long long slots = n - r - 1;
            long long v = v_min;
            while (v < m) {
                const long long choices = m - v;
                const long long c = (slots > 0) ? binom(choices + slots - 1, slots) : 1LL;
                if (c <= k) { k -= c; ++v; } else break;
            }
            out[(long long)r * total + tid] = start + v;
            v_min = v;
        }
    }
}
"""

@lru_cache(maxsize=1)
def _get_kernel() -> cp.RawKernel:
    return cp.RawKernel(_UNRANK_KERNEL_SRC, "unrank_combinations")


def _pass_c_carries(
    starts:      cp.ndarray,
    stops:       cp.ndarray,
    d_parents:   cp.ndarray,
    d_local:     cp.ndarray,
    n:           int,
    replacement: bool,
    total:       int,
) -> List[cp.ndarray]:
    if total == 0 or n == 0:
        return [cp.zeros(0, dtype=np.int64) for _ in range(n)]

    # Two indexed reads — gather parent attributes for every output element
    d_start_vals = starts[d_parents].astype(np.int64)
    d_m_vals     = (stops[d_parents] - starts[d_parents]).astype(np.int64)

    # Single contiguous (n, total) output — avoids n separate allocations
    d_out = cp.empty((n, total), dtype=np.int64, order="C")

    block = 256
    grid  = (int(total) + block - 1) // block

    _get_kernel()(
        (grid,), (block,),
        (
            d_start_vals,
            d_m_vals,
            d_local.astype(np.int64),
            d_out,
            np.int64(total),
            np.int32(n),
            np.int32(int(replacement)),
        ),
    )

    # Return row-views — zero-copy
    return [d_out[r] for r in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def awkward_ListArray_combinations(
    starts:      cp.ndarray,
    stops:       cp.ndarray,
    n:           int,
    replacement: bool = False,
) -> Tuple[List[cp.ndarray], cp.ndarray]:
    """
    Compute all n-combinations for each list described by starts/stops.

    Parameters
    ----------
    starts, stops : 1-D CuPy int arrays of shape (length,)
        Per-list start (inclusive) and stop (exclusive) indices.
    n : int >= 0
        Combination size.
    replacement : bool
        True for combinations with replacement.

    Returns
    -------
    tocarry : list of n CuPy int64 arrays of shape (total,)
    toindex : CuPy int64 array of length n, each element == total
    """
    starts = cp.asarray(starts, dtype=np.int64)
    stops  = cp.asarray(stops,  dtype=np.int64)
    length = int(starts.shape[0])

    if n < 0:
        raise ValueError("n must be >= 0")

    d_offsets = _pass_a_counts(starts, stops, n, replacement)
    total     = int(d_offsets[length].get())   # only host sync in the pipeline

    if total == 0 or n == 0:
        return [cp.zeros(0, dtype=np.int64) for _ in range(n)], cp.zeros(n, dtype=np.int64)

    d_parents, d_local = _pass_b(d_offsets, total)

    tocarry = _pass_c_carries(starts, stops, d_parents, d_local, n, replacement, total)

    return tocarry, cp.full(n, total, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test + benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    print("=" * 60)
    print("awkward_ListArray_combinations — test suite")
    print("=" * 60)

    # Test 1: n=2, no replacement
    starts = cp.array([0, 3], dtype=np.int64)
    stops  = cp.array([3, 5], dtype=np.int64)
    tocarry, toindex = awkward_ListArray_combinations(starts, stops, n=2)
    c0, c1 = tocarry[0].get(), tocarry[1].get()
    print(f"\nTest 1 — n=2, no replacement, total={len(c0)}")
    for i in range(len(c0)):
        print(f"  combo[{i}] = ({c0[i]}, {c1[i]})")
    assert len(c0) == 4
    assert list(c0) == [0, 0, 1, 3], list(c0)
    assert list(c1) == [1, 2, 2, 4], list(c1)
    print("  PASS")

    # Test 2: n=3, no replacement
    starts = cp.array([0], dtype=np.int64)
    stops  = cp.array([4], dtype=np.int64)
    tocarry, _ = awkward_ListArray_combinations(starts, stops, n=3)
    combos = list(zip(tocarry[0].get(), tocarry[1].get(), tocarry[2].get()))
    print(f"\nTest 2 — n=3, no replacement: {combos}")
    assert combos == [(0,1,2),(0,1,3),(0,2,3),(1,2,3)], combos
    print("  PASS")

    # Test 3: n=2, with replacement
    starts = cp.array([0], dtype=np.int64)
    stops  = cp.array([3], dtype=np.int64)
    tocarry, _ = awkward_ListArray_combinations(starts, stops, n=2, replacement=True)
    combos = list(zip(tocarry[0].get(), tocarry[1].get()))
    print(f"\nTest 3 — n=2, with replacement: {combos}")
    assert combos == [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)], combos
    print("  PASS")

    # Test 4: n=0
    starts = cp.array([0, 3], dtype=np.int64)
    stops  = cp.array([3, 5], dtype=np.int64)
    tocarry, _ = awkward_ListArray_combinations(starts, stops, n=0)
    assert len(tocarry) == 0
    print("\nTest 4 — n=0: PASS")

    # Test 5: empty first list
    starts = cp.array([0, 3], dtype=np.int64)
    stops  = cp.array([0, 5], dtype=np.int64)
    tocarry, _ = awkward_ListArray_combinations(starts, stops, n=2)
    c0, c1 = tocarry[0].get(), tocarry[1].get()
    assert list(c0) == [3] and list(c1) == [4], (list(c0), list(c1))
    print("Test 5 — empty first list: PASS")

    # Test 6: large smoke test
    rng    = np.random.default_rng(42)
    length = 100_000
    sizes  = rng.integers(0, 20, size=length).astype(np.int64)
    hs = np.concatenate([[0], np.cumsum(sizes[:-1])])
    he = hs + sizes
    tocarry, toindex = awkward_ListArray_combinations(cp.asarray(hs), cp.asarray(he), n=3)
    total = int(toindex[0].get()) if len(toindex) > 0 else 0
    print(f"\nTest 6 — 100k lists, n=3: total={total:,}  PASS")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)

    # Benchmark
    import time
    print("\nBenchmark: 1M lists, uniform size 10, n=3, no replacement")
    length  = 1_000_000
    sizes2  = np.full(length, 10, dtype=np.int64)
    hs = np.concatenate([[0], np.cumsum(sizes2[:-1])])
    he = hs + sizes2
    ds, de = cp.asarray(hs), cp.asarray(he)
    # warm-up
    awkward_ListArray_combinations(ds, de, n=3)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        awkward_ListArray_combinations(ds, de, n=3)
    cp.cuda.Stream.null.synchronize()
    print(f"  avg wall time: {(time.perf_counter()-t0)/10*1000:.1f} ms")


if __name__ == "__main__":
    _run_tests()
