# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: B023  # closures are consumed within their own loop iteration
"""Profile awkward_reduce_sum: three implementations, head to head.

  A. "previous" — the parents-based compiled kernel from `main`
     (two launches: zero + block-scan with atomicAdd, plus a per-call
     `temp` buffer of lenparents elements), embedded below verbatim.
  B. "offsets .cu" — this branch's rewritten compiled kernel, currently
     *disabled* in dev/generate-kernel-signatures.py (one thread per bin,
     serial loop over the bin).
  C. "cuda.compute" — the active dispatch path (CCCL segmented_reduce).
     Note: it does `input.astype(result.dtype, copy=False)`, which is a
     full input copy whenever the result dtype widens the input (e.g.
     int32 -> int64). The memory profile makes that visible.

All three are called at kernel level with identical device-resident inputs;
`parents` for A is precomputed outside the timed region (in the old pipeline
it already existed). Each config is validated against a CuPy reference
before timing.

Run on a CUDA machine with this checkout + CuPy (+ cuda.compute for C):

    python profile_reduce_sum.py
    python profile_reduce_sum.py --regimes tiny huge

Regimes matter: B is expected to win or tie for many small bins and lose
badly for few huge bins (one thread does the whole bin); A degrades with
atomic contention at small outlength; C should be robust everywhere but
pays JIT on first call and the astype copy on widening dtypes.

NOT for committing — scratch tool for the PR #4056 re-enable/delete decision.
"""

from __future__ import annotations

import argparse
import statistics

import cupy as cp
import numpy as np

PREAMBLE = r"""
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef int int32_t;
typedef unsigned int uint32_t;
#define NO_ERROR 0

// CUDA has no native signed 64-bit atomicAdd; two's complement makes the
// unsigned one equivalent.
__device__ long long atomicAdd(long long* address, long long val) {
  return (long long)atomicAdd((unsigned long long*)address,
                              (unsigned long long)val);
}
"""

# --- A: verbatim from main:src/awkward/_connect/cuda/cuda_kernels/awkward_reduce_sum.cu
OLD_KERNEL = r"""
template <typename T, typename C, typename U, typename V>
__global__ void
awkward_reduce_sum_a(
    T* toptr, const C* fromptr, const U* parents, const V* offsets,
    int64_t lenparents, int64_t outlength, T* temp,
    uint64_t invocation_index, uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < outlength) {
      toptr[thread_id] = 0;
    }
  }
}

template <typename T, typename C, typename U, typename V>
__global__ void
awkward_reduce_sum_b(
    T* toptr, const C* fromptr, const U* parents, const V* offsets,
    int64_t lenparents, int64_t outlength, T* temp,
    uint64_t invocation_index, uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t idx = threadIdx.x;
    int64_t thread_id = blockIdx.x * blockDim.x + idx;

    if (thread_id < lenparents) {
      temp[thread_id] = fromptr[thread_id];
    }
    __syncthreads();

    if (thread_id < lenparents) {
      for (int64_t stride = 1; stride < blockDim.x; stride *= 2) {
        T val = 0;
        if (idx >= stride && thread_id < lenparents &&
            parents[thread_id] == parents[thread_id - stride]) {
          val = temp[thread_id - stride];
        }
        __syncthreads();
        temp[thread_id] += val;
        __syncthreads();
      }

      int64_t parent = parents[thread_id];
      if (idx == blockDim.x - 1 || thread_id == lenparents - 1 ||
          parents[thread_id] != parents[thread_id + 1]) {
        atomicAdd(&toptr[parent], temp[thread_id]);
      }
    }
  }
}
"""

# --- B: verbatim from this branch's awkward_reduce_sum.cu (disabled)
NEW_KERNEL = r"""
template <typename T, typename C, typename V>
__global__ void
awkward_reduce_sum_kernel(
    T* toptr, const C* fromptr, const V* offsets, int64_t outlength,
    uint64_t invocation_index, uint64_t* err_code) {
  if (err_code[0] == NO_ERROR) {
    int64_t bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < outlength) {
      T acc = (T)0;
      int64_t start = (int64_t)offsets[bin];
      int64_t stop  = (int64_t)offsets[bin + 1];
      for (int64_t i = start; i < stop; i++) {
        acc += (T)fromptr[i];
      }
      toptr[bin] = acc;
    }
  }
}
"""

CTYPE = {
    np.dtype(np.int32): "int32_t",
    np.dtype(np.int64): "int64_t",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}

REGIMES = {
    # name: (n_elements, avg_bin_size)
    "tiny": (10_000_000, 4),
    "medium": (10_000_000, 256),
    "huge": (10_000_000, 500_000),
}

DTYPES = [
    (np.float32, np.float32),
    (np.float64, np.float64),
    (np.int64, np.int64),
    (np.int32, np.int64),  # widening: exposes cuda.compute's astype copy
]


class AllocRecorder(cp.cuda.MemoryHook):
    name = "reduce-sum-profile"

    def __init__(self):
        self.sizes = []

    def malloc_postprocess(self, **kwargs):
        self.sizes.append(kwargs.get("mem_size", kwargs.get("size", 0)))


def build_modules(from_dtype, to_dtype):
    fc, tc = CTYPE[np.dtype(from_dtype)], CTYPE[np.dtype(to_dtype)]
    old_names = [
        f"awkward_reduce_sum_a<{tc}, {fc}, int64_t, int64_t>",
        f"awkward_reduce_sum_b<{tc}, {fc}, int64_t, int64_t>",
    ]
    new_names = [f"awkward_reduce_sum_kernel<{tc}, {fc}, int64_t>"]
    old_mod = cp.RawModule(
        code=PREAMBLE + OLD_KERNEL, options=("-std=c++14",), name_expressions=old_names
    )
    new_mod = cp.RawModule(
        code=PREAMBLE + NEW_KERNEL, options=("-std=c++14",), name_expressions=new_names
    )
    return (
        old_mod.get_function(old_names[0]),
        old_mod.get_function(old_names[1]),
        new_mod.get_function(new_names[0]),
    )


def make_input(n_elements, avg_bin, from_dtype, seed=0):
    rng = np.random.default_rng(seed)
    n_bins = max(1, n_elements // avg_bin)
    counts = rng.poisson(avg_bin, n_bins).astype(np.int64)
    offsets = np.zeros(n_bins + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    if np.dtype(from_dtype).kind == "f":
        data = rng.standard_normal(total).astype(from_dtype)
    else:
        data = rng.integers(-100, 100, total).astype(from_dtype)
    d_data = cp.asarray(data)
    d_offsets = cp.asarray(offsets)
    # parents for the old kernel: bin id per element (precomputed, untimed —
    # the old pipeline already had it on hand)
    d_parents = cp.searchsorted(
        d_offsets[1:], cp.arange(total, dtype=cp.int64), side="right"
    ).astype(cp.int64)
    return d_data, d_offsets, d_parents, n_bins, total


def gpu_time_ms(fn):
    cp.cuda.Device().synchronize()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    fn()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop)


def bench(fn, runs=11):
    first = gpu_time_ms(fn)
    return first, statistics.median(gpu_time_ms(fn) for _ in range(runs))


def allocs(fn):
    fn()
    cp.cuda.Device().synchronize()
    rec = AllocRecorder()
    with rec:
        fn()
        cp.cuda.Device().synchronize()
    return len(rec.sizes), sum(rec.sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES))
    parser.add_argument("--elements", type=int, default=None)
    args = parser.parse_args()

    try:
        from awkward._connect.cuda import _compute

        have_compute = True
    except Exception as err:
        print(f"cuda.compute unavailable ({err}); skipping impl C")
        have_compute = False

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    print(f"device: {props['name'].decode()}")

    err_code = cp.zeros(1, dtype=cp.uint64)

    for regime in args.regimes:
        n_elements, avg_bin = REGIMES[regime]
        if args.elements:
            n_elements = args.elements
        for from_dt, to_dt in DTYPES:
            data, offsets, parents, n_bins, total = make_input(
                n_elements, avg_bin, from_dt
            )
            out_a = cp.empty(n_bins, dtype=to_dt)
            out_b = cp.empty(n_bins, dtype=to_dt)
            out_c = cp.empty(n_bins, dtype=to_dt)
            kern_a0, kern_a1, kern_b = build_modules(from_dt, to_dt)

            block = min(total, 1024)
            grid_ab = (max(total, n_bins) + block - 1) // block if block else 1
            grid_bins = (n_bins + block - 1) // block if block else 1

            def run_old():
                temp = cp.zeros(total, dtype=to_dt)  # per-call, as on main
                kern_a0(
                    (grid_ab,),
                    (block,),
                    (
                        out_a,
                        data,
                        parents,
                        offsets,
                        np.int64(total),
                        np.int64(n_bins),
                        temp,
                        np.uint64(0),
                        err_code,
                    ),
                )
                kern_a1(
                    (grid_ab,),
                    (block,),
                    (
                        out_a,
                        data,
                        parents,
                        offsets,
                        np.int64(total),
                        np.int64(n_bins),
                        temp,
                        np.uint64(0),
                        err_code,
                    ),
                )

            def run_new():
                kern_b(
                    (grid_bins,),
                    (block,),
                    (out_b, data, offsets, np.int64(n_bins), np.uint64(0), err_code),
                )

            def run_compute():
                _compute.awkward_reduce_sum(out_c, data, offsets, n_bins)

            impls = [("A previous/parents", run_old), ("B offsets .cu", run_new)]
            if have_compute:
                impls.append(("C cuda.compute", run_compute))

            # correctness first (cumsum-based segmented sums)
            csum = cp.zeros(total + 1, dtype=to_dt)
            cp.cumsum(data.astype(to_dt), out=csum[1:])
            reference = csum[offsets[1:]] - csum[offsets[:-1]]
            for name, fn in impls:
                fn()
                cp.cuda.Device().synchronize()
                got = {"A": out_a, "B": out_b, "C": out_c}[name[0]]
                if np.dtype(to_dt).kind == "f":
                    ok = bool(cp.allclose(got, reference, rtol=1e-4, atol=1e-6))
                else:
                    ok = bool((got == reference).all())
                if not ok:
                    print(
                        f"!! {regime} {from_dt.__name__}->{to_dt.__name__} {name}: WRONG RESULT"
                    )

            in_bytes = total * np.dtype(from_dt).itemsize
            print(
                f"\n--- {regime}: {total:,} elems, {n_bins:,} bins, "
                f"{from_dt.__name__}->{to_dt.__name__} ({in_bytes / 1e6:.0f} MB in)"
            )
            for name, fn in impls:
                first, steady = bench(fn)
                gbps = in_bytes / (steady * 1e-3) / 1e9
                n_alloc, alloc_bytes = allocs(fn)
                print(
                    f"  {name:<20} first {first:8.3f} ms | steady {steady:8.3f} ms "
                    f"| {gbps:7.1f} GB/s | {n_alloc} allocs {alloc_bytes / 1e6:8.2f} MB"
                )

    print(
        "\nReading the results:\n"
        "  * A's alloc column shows the per-call lenparents `temp` buffer; its\n"
        "    GB/s fades at small outlength (atomic contention) and it pays two\n"
        "    launches + a parents-sized scan regardless of regime.\n"
        "  * B should lead in 'tiny', collapse in 'huge' (single thread per bin\n"
        "    -> utilization ~ n_bins/SM count). If B only loses in 'huge',\n"
        "    a grid-stride or warp-per-bin variant would fix it.\n"
        "  * C's alloc column shows the astype copy for int32->int64; compare\n"
        "    its int64->int64 row to see the copy's cost. First-call vs steady\n"
        "    gap is the cuda.compute JIT.\n"
    )


if __name__ == "__main__":
    main()
