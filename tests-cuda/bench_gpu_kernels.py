# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E731
"""GPU micro-benchmarks + memory profile for four suspected cuda.compute bottlenecks:

  1. awkward_missing_repeat            (constant / per-call overhead)
  2. awkward_reduce_sum_complex,
     awkward_reduce_prod_complex       ("compute-bound" complex math claim)
  3. awkward_reduce_sum_bool_complex   (branch divergence claim vs. intermediate buffer)
  4. *_rpad_and_clip_axis1             (axis-1 scaling claim)

Run on a machine with CUDA + CuPy + this checkout installed:

    python bench_gpu_kernels.py                  # everything
    python bench_gpu_kernels.py --filter rpad    # one group
    python bench_gpu_kernels.py --no-mem         # skip memory profiling

For each benchmark this reports:
  * first-call time (includes cuda.compute op JIT) vs. steady-state median
    -> a large ratio that persists across *calls with fresh arrays* indicates
       the op cache is being defeated (e.g. closures capturing device arrays)
  * per-call overhead on tiny inputs (the constant-overhead claim)
  * a size sweep with effective GB/s against ideal bytes moved
    -> memory-bound kernels should plateau near device bandwidth; a plateau
       far below it (with allocations, below) points at intermediate buffers
  * allocations made during one steady-state call (count + bytes vs. ideal)
    -> e.g. sum_bool_complex's un-fused `mapped_data` shows up here

NOT for committing — scratch tool for the PR #4056 re-enable/delete decision.
"""

from __future__ import annotations

import argparse
import statistics

import cupy as cp
import numpy as np

import awkward as ak

# ---------------------------------------------------------------- utilities


class AllocRecorder(cp.cuda.MemoryHook):
    """Records every pool allocation made inside a `with` block."""

    name = "awkward-bench-alloc-recorder"

    def __init__(self):
        self.sizes = []

    def malloc_postprocess(self, **kwargs):
        # mem_size = rounded size actually taken from the pool
        self.sizes.append(kwargs.get("mem_size", kwargs.get("size", 0)))

    @property
    def total(self):
        return sum(self.sizes)


def gpu_time_ms(fn, *, sync_before=True):
    """Time one call with CUDA events (ms)."""
    if sync_before:
        cp.cuda.Device().synchronize()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    fn()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop)


def bench(fn, *, steady_runs=11):
    """Returns (first_ms, steady_median_ms)."""
    first = gpu_time_ms(fn)
    times = [gpu_time_ms(fn) for _ in range(steady_runs)]
    return first, statistics.median(times)


def per_call_overhead_us(fn, *, calls=200):
    """Mean per-call wall time (µs) for many repeated tiny calls."""
    cp.cuda.Device().synchronize()
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for _ in range(calls):
        fn()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) / calls * 1e3


def mem_profile(fn):
    """Allocations during one call: (n_allocs, total_bytes)."""
    cp.cuda.Device().synchronize()
    rec = AllocRecorder()
    with rec:
        fn()
        cp.cuda.Device().synchronize()
    return len(rec.sizes), rec.total


KERNEL_CALLS: set[str] = set()


def install_kernel_trace():
    """Record which awkward kernels each benchmark actually dispatches."""
    try:
        from awkward._backends.cupy import CupyBackend

        original = CupyBackend.__getitem__

        def traced(self, key):
            KERNEL_CALLS.add(key[0] if isinstance(key, tuple) else key)
            return original(self, key)

        CupyBackend.__getitem__ = traced
    except Exception as err:  # pragma: no cover - best effort
        print(f"(kernel tracing unavailable: {err})")


def traced_kernels(fn):
    KERNEL_CALLS.clear()
    fn()
    cp.cuda.Device().synchronize()
    return sorted(KERNEL_CALLS)


def report(name, fn, *, ideal_bytes, do_mem=True, tiny_fn=None):
    kernels = ", ".join(k for k in traced_kernels(fn) if "awkward" in k) or "?"
    first, steady = bench(fn)
    gbps = ideal_bytes / (steady * 1e-3) / 1e9 if steady > 0 else float("nan")
    print(f"\n=== {name}")
    print(f"    kernels hit       : {kernels}")
    print(
        f"    first call        : {first:8.3f} ms   (steady x{first / steady:5.1f})"
        if steady > 0
        else f"    first call        : {first:8.3f} ms"
    )
    print(f"    steady median     : {steady:8.3f} ms   ({gbps:7.1f} GB/s effective)")
    if tiny_fn is not None:
        print(
            f"    tiny-input/call   : {per_call_overhead_us(tiny_fn):8.1f} us  (constant overhead)"
        )
    if do_mem:
        n, total = mem_profile(fn)
        ratio = total / ideal_bytes if ideal_bytes else float("nan")
        print(
            f"    allocs/steady call: {n} allocs, {total / 1e6:9.2f} MB"
            f"  ({ratio:4.1f}x ideal output -> intermediates if >> 1)"
        )


# ---------------------------------------------------------------- inputs


def ragged_complex(n_elements, avg_list=64, dtype=np.complex128, seed=0):
    rng = np.random.default_rng(seed)
    n_rows = max(1, n_elements // avg_list)
    counts = rng.poisson(avg_list, n_rows)
    total = int(counts.sum())
    data = rng.standard_normal(total) + 1j * rng.standard_normal(total)
    arr = ak.unflatten(ak.Array(data.astype(dtype)), counts)
    return ak.to_backend(arr, "cuda")


def ragged_float(n_elements, avg_list=64, seed=0):
    rng = np.random.default_rng(seed)
    n_rows = max(1, n_elements // avg_list)
    counts = rng.poisson(avg_list, n_rows)
    data = rng.standard_normal(int(counts.sum()))
    return ak.to_backend(ak.unflatten(ak.Array(data), counts), "cuda")


def regular_2d(n_rows, n_cols, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_cols))
    return ak.to_backend(ak.Array(data), "cuda")


# ---------------------------------------------------------------- benchmarks


def bench_missing_repeat(sizes, do_mem):
    # arr[:, [0, None, cols-1]] on a regular 2D array goes through
    # _getitem_next_missing -> awkward_missing_repeat (repetitions = n_rows).
    for n in sizes:
        rows, cols = max(1, n // 16), 16
        arr = regular_2d(rows, cols)
        slicer = [0, None, cols - 1]
        fn = lambda arr=arr, slicer=slicer: arr[:, slicer]
        tiny = regular_2d(8, 8)
        tiny_fn = lambda tiny=tiny: tiny[:, [0, None, 7]]
        ideal = rows * 3 * 8 + 3 * 8  # outindex write + index read
        report(
            f"missing_repeat  rows={rows:>9,}",
            fn,
            ideal_bytes=ideal,
            do_mem=do_mem,
            tiny_fn=tiny_fn,
        )


def bench_complex_reducers(sizes, do_mem):
    for n in sizes:
        arr = ragged_complex(n)
        n_rows = len(arr)
        itemsize = 16
        ideal = n * itemsize + n_rows * itemsize
        tiny = ragged_complex(64, avg_list=8)
        for name, op in (("sum", ak.sum), ("prod", ak.prod)):
            fn = lambda op=op, arr=arr: op(arr, axis=1)
            tiny_fn = lambda op=op, tiny=tiny: op(tiny, axis=1)
            report(
                f"{name}_complex  N={n:>10,}",
                fn,
                ideal_bytes=ideal,
                do_mem=do_mem,
                tiny_fn=tiny_fn,
            )


def bench_sum_bool_complex(sizes, do_mem):
    # ak.any on complex -> awkward_reduce_sum_bool_complex.
    # Current impl materializes an int8 intermediate of N elements; the
    # alloc line below makes that visible (expect ~N bytes beyond output).
    for n in sizes:
        arr = ragged_complex(n)
        n_rows = len(arr)
        ideal = n * 16 + n_rows  # read input + write bool output
        tiny = ragged_complex(64, avg_list=8)
        fn = lambda arr=arr: ak.any(arr, axis=1)
        tiny_fn = lambda tiny=tiny: ak.any(tiny, axis=1)
        report(
            f"sum_bool_complex (ak.any)  N={n:>10,}",
            fn,
            ideal_bytes=ideal,
            do_mem=do_mem,
            tiny_fn=tiny_fn,
        )


def bench_rpad_and_clip(sizes, do_mem):
    # axis-1 scaling: fix elements, sweep target (the "scaling" axis).
    for n in sizes:
        arr = ragged_float(n)  # ListOffsetArray path
        reg = regular_2d(max(1, n // 64), 64)  # RegularArray path
        for target in (2, 64, 256):
            ideal = len(arr) * target * 8
            fn = lambda t=target, arr=arr: ak.pad_none(arr, t, axis=1, clip=True)
            report(
                f"rpad_and_clip ListOffset  N={n:>10,} target={target:>4}",
                fn,
                ideal_bytes=ideal,
                do_mem=do_mem,
            )
        ideal = len(reg) * 256 * 8
        fn = lambda reg=reg: ak.pad_none(reg, 256, axis=1, clip=True)
        tiny = regular_2d(8, 8)
        tiny_fn = lambda tiny=tiny: ak.pad_none(tiny, 4, axis=1, clip=True)
        report(
            f"rpad_and_clip Regular     N={n:>10,} target= 256",
            fn,
            ideal_bytes=ideal,
            do_mem=do_mem,
            tiny_fn=tiny_fn,
        )


GROUPS = {
    "missing_repeat": bench_missing_repeat,
    "complex": bench_complex_reducers,
    "sum_bool_complex": bench_sum_bool_complex,
    "rpad": bench_rpad_and_clip,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10_000, 1_000_000, 10_000_000],
        help="approximate element counts to sweep",
    )
    parser.add_argument("--filter", default="", help="substring of group name")
    parser.add_argument("--no-mem", action="store_true", help="skip memory profiling")
    args = parser.parse_args()

    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    bw = 2 * props["memoryClockRate"] * 1e3 * (props["memoryBusWidth"] // 8) / 1e9
    print(f"device: {props['name'].decode()}  (theoretical ~{bw:.0f} GB/s)")
    print(f"awkward {ak.__version__}, cupy {cp.__version__}")

    install_kernel_trace()

    for name, fn in GROUPS.items():
        if args.filter and args.filter not in name:
            continue
        print(f"\n############ {name} ############")
        fn(args.sizes, not args.no_mem)

    print(
        "\nReading the results:\n"
        "  * 'tiny-input/call' >> steady-per-element cost -> constant overhead "
        "(claim 1); compare against a re-enabled .cu kernel for the same op\n"
        "  * complex sum/prod GB/s close to other reducers -> memory-bound, "
        "not compute-bound (claim 2)\n"
        "  * sum_bool_complex allocs >> ideal -> intermediate buffer, fuse "
        "with TransformIterator; branch divergence is not the issue (claim 3)\n"
        "  * rpad GB/s flat across target values -> linear scaling, no axis-1 "
        "pathology (claim 4)\n"
    )


if __name__ == "__main__":
    main()
