# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: T201, TID251
"""Profile the CPU awkward_reduce_sum: this branch vs. the previous (main).

  A. "previous" — main's parents-based kernel, verbatim:
       memset(toptr); for i: toptr[parents[i]] += fromptr[i]
     A serial scatter with an indirect store per element; not vectorizable.
     `parents` is precomputed outside the timed region (the old pipeline
     already had it on hand), so the comparison is kernel-only and favors A.
  B. "offsets" — this branch's kernel, verbatim: per-bin loop, 4-way
     unrolled accumulators, __restrict__; compiled twice, with and without
     OpenMP (the pragma gates on outlength > 1024).

Both are compiled at runtime with the same compiler and -O3, validated
against NumPy, then timed over three bin-size regimes and four dtype
combinations. GB/s is computed against input bytes only.

    python profile_reduce_sum_cpu.py
    python profile_reduce_sum_cpu.py --elements 50000000 --runs 9

NOT for committing — scratch tool for PR #4056.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import subprocess
import tempfile
import time

import numpy as np

SOURCE = r"""
#include <cstdint>
#include <cstring>

// ---- A: previous implementation (main), verbatim apart from the name -----
template <typename OUT, typename IN>
void old_reduce_sum(
  OUT* toptr, const IN* fromptr, const int64_t* parents,
  int64_t lenparents, int64_t outlength) {
  std::memset(toptr, 0, outlength * sizeof(OUT));
  for (int64_t i = 0; i < lenparents; i++) {
    int64_t parent_idx = parents[i];
    toptr[parent_idx] += static_cast<OUT>(fromptr[i]);
  }
}

// ---- B: this branch's implementation, verbatim apart from the name -------
template <typename OUT, typename IN>
void new_reduce_sum(
  OUT* __restrict__ toptr, const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets, int64_t outlength) {
  #ifdef _OPENMP
  #pragma omp parallel for if(outlength > 1024) schedule(static)
  #endif
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop  = offsets[bin + 1];
    OUT a0 = OUT{}, a1 = OUT{}, a2 = OUT{}, a3 = OUT{};
    int64_t i = start;
    for (; i + 4 <= stop; i += 4) {
      a0 += static_cast<OUT>(fromptr[i + 0]);
      a1 += static_cast<OUT>(fromptr[i + 1]);
      a2 += static_cast<OUT>(fromptr[i + 2]);
      a3 += static_cast<OUT>(fromptr[i + 3]);
    }
    OUT acc = (a0 + a1) + (a2 + a3);
    for (; i < stop; i++) {
      acc += static_cast<OUT>(fromptr[i]);
    }
    toptr[bin] = acc;
  }
}

#define EXPORT(OUT_T, IN_T, NAME)                                            \
  extern "C" void old_##NAME(OUT_T* t, const IN_T* f, const int64_t* p,      \
                             int64_t lp, int64_t ol) {                       \
    old_reduce_sum<OUT_T, IN_T>(t, f, p, lp, ol);                            \
  }                                                                          \
  extern "C" void new_##NAME(OUT_T* t, const IN_T* f, const int64_t* o,      \
                             int64_t ol) {                                   \
    new_reduce_sum<OUT_T, IN_T>(t, f, o, ol);                                \
  }

EXPORT(float,   float,   f32_f32)
EXPORT(double,  double,  f64_f64)
EXPORT(int64_t, int64_t, i64_i64)
EXPORT(int64_t, int32_t, i64_i32)
"""

COMBOS = [
    ("f32_f32", np.float32, np.float32),
    ("f64_f64", np.float64, np.float64),
    ("i64_i64", np.int64, np.int64),
    ("i64_i32", np.int32, np.int64),  # widening, like ak.sum on int32
]

REGIMES = {
    "tiny": 4,
    "medium": 256,
    "huge": 500_000,
}


def compile_lib(openmp):
    src = tempfile.NamedTemporaryFile(suffix=".cpp", delete=False, mode="w")
    src.write(SOURCE)
    src.close()
    out = src.name.replace(".cpp", "_omp.so" if openmp else ".so")
    cmd = ["g++", "-O3", "-shared", "-fPIC", "-o", out, src.name]
    if openmp:
        cmd.insert(2, "-fopenmp")
    subprocess.run(cmd, check=True)
    return ctypes.CDLL(out)


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
    parents = np.repeat(np.arange(n_bins, dtype=np.int64), counts)
    return data, offsets, parents, n_bins, total


def timeit(fn, runs):
    fn()  # warm-up (page faults, OMP thread pool spin-up)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3  # ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=20_000_000)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    lib_serial = compile_lib(openmp=False)
    lib_omp = compile_lib(openmp=True)

    print(f"cores: {os.cpu_count()}, elements per regime: ~{args.elements:,}")
    header = (
        f"{'regime':<8} {'dtype':<9} {'bins':>10} | "
        f"{'A prev (ms)':>12} {'B serial':>10} {'B omp':>10} | "
        f"{'B/A speedup':>11} {'omp gain':>9}"
    )
    print(header)
    print("-" * len(header))

    for regime, avg_bin in REGIMES.items():
        for name, from_dt, to_dt in COMBOS:
            data, offsets, parents, n_bins, total = make_input(
                args.elements, avg_bin, from_dt
            )
            out_a = np.empty(n_bins, dtype=to_dt)
            out_b = np.empty(n_bins, dtype=to_dt)
            out_c = np.empty(n_bins, dtype=to_dt)

            c_t = np.ctypeslib.ndpointer(dtype=to_dt)
            c_f = np.ctypeslib.ndpointer(dtype=from_dt)
            c_i = np.ctypeslib.ndpointer(dtype=np.int64)

            old = getattr(lib_serial, f"old_{name}")
            old.argtypes = [c_t, c_f, c_i, ctypes.c_int64, ctypes.c_int64]
            new_s = getattr(lib_serial, f"new_{name}")
            new_s.argtypes = [c_t, c_f, c_i, ctypes.c_int64]
            new_p = getattr(lib_omp, f"new_{name}")
            new_p.argtypes = [c_t, c_f, c_i, ctypes.c_int64]

            run_a = lambda: old(out_a, data, parents, total, n_bins)  # noqa: E731,B023
            run_b = lambda: new_s(out_b, data, offsets, n_bins)  # noqa: E731,B023
            run_c = lambda: new_p(out_c, data, offsets, n_bins)  # noqa: E731,B023

            # correctness (cumsum-based segmented reference)
            csum = np.zeros(total + 1, dtype=np.float64 if to_dt(0).dtype.kind == "f" else np.int64)
            np.cumsum(data.astype(csum.dtype), out=csum[1:])
            reference = (csum[offsets[1:]] - csum[offsets[:-1]]).astype(to_dt)
            for run, out in ((run_a, out_a), (run_b, out_b), (run_c, out_c)):
                run()
                if np.dtype(to_dt).kind == "f":
                    assert np.allclose(out, reference, rtol=1e-3, atol=1e-3)
                else:
                    assert (out == reference).all()

            t_a = timeit(run_a, args.runs)
            t_b = timeit(run_b, args.runs)
            t_c = timeit(run_c, args.runs)

            in_gb = total * np.dtype(from_dt).itemsize / 1e9
            print(
                f"{regime:<8} {name:<9} {n_bins:>10,} | "
                f"{t_a:>9.2f} ({in_gb / (t_a * 1e-3):4.1f}GB/s) "
                f"{t_b:>10.2f} {t_c:>10.2f} | "
                f"{t_a / t_b:>10.2f}x {t_b / t_c:>8.2f}x"
            )

    print(
        "\nNotes: A includes its memset but NOT the cost of building `parents`\n"
        "(8 bytes/element extra bandwidth in the old pipeline, plus the kernel\n"
        "that produced it). B omp uses all cores when bins > 1024, serial\n"
        "otherwise — the 'huge' regime shows the gate keeping it serial."
    )


if __name__ == "__main__":
    main()
