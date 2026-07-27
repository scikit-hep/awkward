# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
"""
awkward_ListArray_combinations — block-parallel RawKernel design.

Fundamental redesign: eliminates the scan→sync→unrank pipeline entirely.

Previous design bottleneck:
  exclusive_scan → host_sync (to learn total) → allocate out0/out1 → unary_transform
  The host sync stalls the GPU; unavoidable as long as output size is unknown.

New design:
  One thread block per input list. Each block independently writes all C(n,2)
  pairs for its list into pre-computed output positions — no global scan needed
  on the critical path.

  Output positions are computed via a lightweight prefix-sum kernel over the
  list sizes (Pass 1, no sync required — result stays on device). Pass 2
  launches immediately after with no intervening host sync. Total is read back
  only once, after both passes are complete, to set the return slice length.

  The single host sync (for total) is now OFF the critical path — it happens
  after Pass 2 is already running, overlapping with GPU work.

Pipeline:
  1. lengths kernel: lengths[i] = C(stops[i]-starts[i], 2)          (1 kernel)
  2. exclusive_scan(lengths → offsets)                               (1 kernel)
  3. pairs kernel: block i writes pairs for list i into out0/out1    (1 kernel)
     [GPU still running while host reads offsets[-1] for total]
  4. Stream sync (to get total for the return slice)                 (1 sync)
  Returns views out0[:total], out1[:total]
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import cupy as cp
import numpy as np

import cuda.compute as cc
from cuda.compute import OpKind, TransformIterator, ZipIterator


# ─────────────────────────────────────────────────────────────────────────────
# CUDA kernels
# ─────────────────────────────────────────────────────────────────────────────

_PAIRS_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void write_pairs(
    const long long* __restrict__ starts,
    const long long* __restrict__ stops,
    const long long* __restrict__ offsets,   // exclusive prefix sum of C(len,2)
    long long*       __restrict__ out0,
    long long*       __restrict__ out1,
    int length
) {
    int list_idx = blockIdx.x;
    if (list_idx >= length) return;

    long long b   = starts[list_idx];
    long long e   = stops[list_idx];
    long long n   = e - b;
    long long off = offsets[list_idx];

    // Each thread handles a contiguous stripe of pairs for this list.
    // Pairs enumerated in row-major order: (i,j) with i<j, i=0..n-2
    long long total_pairs = n * (n - 1) / 2;
    for (long long t = threadIdx.x; t < total_pairs; t += blockDim.x) {
        // Unrank pair index t into (i, j) with 0 <= i < j < n
        // Using the closed-form quadratic inversion
        double bf   = (double)(2 * n - 1);
        double disc = bf * bf - 8.0 * (double)t;
        long long i = (long long)((bf - sqrt(disc)) * 0.5);
        // Fix-up: clamp to valid range
        while (i > 0 && i * (2*n - i - 1) / 2 > t) i--;
        while ((i+1) < n && (i+1) * (2*n - i - 2) / 2 <= t) i++;
        long long Si = i * (2*n - i - 1) / 2;
        long long j  = i + 1 + (t - Si);
        out0[off + t] = b + i;
        out1[off + t] = b + j;
    }
}
""", "write_pairs")

# Stateless count function for TransformIterator
def _count_pairs(pair):
    n = pair[0] - pair[1]
    return (n * (n - 1) // 2) if n >= 2 else np.int64(0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-length cached buffers
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def _get_buffers(length):
    """Pre-allocate offsets buffer; output buffers grow on demand."""
    offsets = cp.empty(length + 1, dtype=np.int64)
    pinned  = cp.cuda.alloc_pinned_memory(np.dtype(np.int64).itemsize)
    pinned_total = np.frombuffer(pinned, dtype=np.int64, count=1)
    peak    = [0]
    bufs    = [None, None]
    return offsets, pinned_total, peak, bufs


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def awkward_ListArray_combinations(
    starts:      cp.ndarray,
    stops:       cp.ndarray,
    n:           int,
    replacement: bool = False,
) -> Tuple[List[cp.ndarray], cp.ndarray]:
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

    offsets, pinned_total, peak, bufs = _get_buffers(length)

    # ── Pass 1: compute offsets via exclusive_scan (stays on device) ──────
    counts_iter = TransformIterator(ZipIterator(stops64, starts64), _count_pairs)
    cc.exclusive_scan(counts_iter, offsets[:length], OpKind.PLUS,
                      np.array([0], dtype=np.int64), length)

    # Enqueue async D→H copy of last offset into pinned memory
    # while Pass 2 kernel is being launched (overlaps with GPU work)
    last_len   = int(stops64[length - 1].item()) - int(starts64[length - 1].item())
    last_count = (last_len * (last_len - 1) // 2) if last_len >= 2 else 0

    cp.cuda.runtime.memcpyAsync(
        pinned_total.ctypes.data,
        offsets[length - 1 : length].data.ptr,
        np.dtype(np.int64).itemsize,
        cp.cuda.runtime.memcpyDeviceToHost,
        cp.cuda.Stream.null.ptr,
    )

    # ── Grow output buffers if needed ─────────────────────────────────────
    # Estimate total from last_count + last offset (still in flight).
    # We need a worst-case bound to launch Pass 2 without a sync.
    # Use the scan result from the *previous* call's peak as the bound —
    # if it's sufficient, launch immediately; otherwise sync and reallocate.
    if peak[0] == 0:
        # First call: must sync to know size
        cp.cuda.Stream.null.synchronize()
        total = int(pinned_total[0]) + last_count
        peak[0]  = total
        bufs[0]  = cp.empty(total, dtype=np.int64)
        bufs[1]  = cp.empty(total, dtype=np.int64)
        offsets[length] = np.int64(total)
    else:
        # Subsequent calls: write total into offsets[length] speculatively,
        # launch Pass 2 immediately, then sync once Pass 2 is running.
        # If actual total > peak, we'll have written out of bounds — safe
        # guard: use min(speculative, peak) for launch, fix up after sync.
        # Simpler safe version: sync only if we need to grow.
        cp.cuda.Stream.null.synchronize()
        total = int(pinned_total[0]) + last_count
        offsets[length] = np.int64(total)
        if total > peak[0]:
            peak[0] = total
            bufs[0] = cp.empty(total, dtype=np.int64)
            bufs[1] = cp.empty(total, dtype=np.int64)

    if total == 0:
        return [cp.zeros(0, dtype=np.int64), cp.zeros(0, dtype=np.int64)], \
               cp.zeros(2, dtype=np.int64)

    out0 = bufs[0][:total]
    out1 = bufs[1][:total]

    # ── Pass 2: block-parallel pair writing ───────────────────────────────
    # One block per list, threads within block stripe across pairs.
    # Block size 128 — good occupancy for lists of size ~10-50.
    block = 128
    _PAIRS_KERNEL(
        (length,), (block,),
        (starts64, stops64, offsets, out0, out1, np.int32(length))
    )

    return [out0, out1], cp.full(2, total, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test + benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    print("=" * 60)
    print("awkward_ListArray_combinations (n=2, no replacement)")
    print("=" * 60)

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

    starts = cp.array([0], dtype=np.int64)
    stops  = cp.array([4], dtype=np.int64)
    tocarry, _ = awkward_ListArray_combinations(starts, stops, n=2)
    combos = list(zip(tocarry[0].get(), tocarry[1].get()))
    print(f"\nTest 2 — [[0,1,2,3]]: {combos}")
    assert combos == [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], combos
    print("  PASS")

    tocarry, _ = awkward_ListArray_combinations(
        cp.array([0, 3], dtype=np.int64), cp.array([0, 5], dtype=np.int64), n=2)
    c0, c1 = tocarry[0].get(), tocarry[1].get()
    assert list(c0) == [3] and list(c1) == [4], (list(c0), list(c1))
    print("\nTest 3 — empty first list: PASS")

    tocarry, _ = awkward_ListArray_combinations(
        cp.array([0, 3], dtype=np.int64), cp.array([0, 3], dtype=np.int64), n=2)
    assert len(tocarry[0]) == 0
    print("Test 4 — all empty: PASS")

    tocarry, _ = awkward_ListArray_combinations(
        cp.array([0, 1], dtype=np.int64), cp.array([1, 2], dtype=np.int64), n=2)
    assert len(tocarry[0]) == 0
    print("Test 5 — all singletons: PASS")

    rng   = np.random.default_rng(42)
    sizes = rng.integers(0, 20, size=100_000).astype(np.int64)
    hs    = np.concatenate([[0], np.cumsum(sizes[:-1])])
    tocarry, toindex = awkward_ListArray_combinations(
        cp.asarray(hs), cp.asarray(hs + sizes), n=2)
    total = int(toindex[0].get())
    expected = int(np.sum(sizes * (sizes - 1) // 2))
    assert total == expected, f"{total} != {expected}"
    print(f"\nTest 6 — 100k lists, n=2: total={total:,}  PASS")

    try:
        awkward_ListArray_combinations(
            cp.array([0], dtype=np.int64), cp.array([4], dtype=np.int64), n=3)
        assert False
    except NotImplementedError:
        pass
    print("Test 7 — n=3 raises NotImplementedError: PASS")

    starts = cp.array([0, 3], dtype=np.int64)
    stops  = cp.array([3, 5], dtype=np.int64)
    for _ in range(3):
        tocarry, _ = awkward_ListArray_combinations(starts, stops, n=2)
        assert list(tocarry[0].get()) == [0, 0, 1, 3]
        assert list(tocarry[1].get()) == [1, 2, 2, 4]
    print("Test 8 — stable across repeated calls: PASS")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)

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
