"""CPU reduction/sort benchmark targeting the parents->offsets kernel migration.

Records, per op:
  * best-of-N wall time (in-process), and
  * peak resident-memory ("peak_mb") measured in a COLD subprocess: a fresh
    interpreter builds just that one array and runs the op once, and we report
    the kernel-tracked high-water mark (resource.ru_maxrss) gained during the
    op above the post-build resident set. Running cold avoids the warm-heap /
    page-retention problem that makes in-process RSS sampling read ~0.

The op array is rebuilt deterministically (seed 12345) in the child, so the
pip and branch runs measure identical data and their peak_mb is comparable.

Usage:
    python bench_ak_funcs_cpu.py [out.json] [--no-mem]
    # (internal) python bench_ak_funcs_cpu.py --mem-child <scale> <op> <dtype>
"""

from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time

import numpy as np

import awkward as ak

print(
    "awkward",
    ak.__version__,
    "| awkward_cpp",
    getattr(__import__("awkward_cpp"), "__version__", "?"),
)

rng = np.random.default_rng(12345)


def make_ragged(n_lists, avg_len, dtype):
    counts = rng.integers(0, 2 * avg_len + 1, size=n_lists)
    total = int(counts.sum())
    if dtype is bool:
        data = rng.integers(0, 2, size=total).astype(bool)
    elif dtype is np.complex128:
        data = (rng.standard_normal(total) + 1j * rng.standard_normal(total)).astype(
            np.complex128
        )
    elif np.issubdtype(dtype, np.integer):
        data = rng.integers(-1000, 1000, size=total).astype(dtype)
    else:
        data = rng.standard_normal(total).astype(dtype)
    offsets = np.zeros(n_lists + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(offsets), ak.contents.NumpyArray(data)
    )
    return ak.Array(layout)


def bench(fn, arr, repeat=7, inner=3):
    fn(arr)  # warmup
    best = float("inf")
    for _ in range(repeat):
        gc.disable()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn(arr)
        t1 = time.perf_counter()
        gc.enable()
        best = min(best, (t1 - t0) / inner)
    return best


def _ru_maxrss_bytes():
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kibibytes.
    return m if sys.platform == "darwin" else m * 1024


def _dtype_from_name(dtn):
    if dtn == "bool":
        return bool
    if dtn == "complex128":
        return np.complex128
    return getattr(np, dtn)


# scales: (n_lists, avg_len)
SCALES = {
    "few_long": (1_000, 10_000),  # few segments, long lists
    "many_short": (2_000_000, 5),  # many segments, short lists
    "balanced": (200_000, 50),  # balanced
}

OPS = []


def reg(name, fn, dtypes):
    OPS.append((name, fn, dtypes))


reg("sum_axis1", lambda a: ak.sum(a, axis=1), [np.int64, np.float64])
reg("sum_axisNone", lambda a: ak.sum(a, axis=None), [np.int64, np.float64])
reg("prod_axis1", lambda a: ak.prod(a, axis=1), [np.int64, np.float64])
reg("max_axis1", lambda a: ak.max(a, axis=1), [np.int64, np.float64])
reg("min_axis1", lambda a: ak.min(a, axis=1), [np.int64, np.float64])
reg("argmax_axis1", lambda a: ak.argmax(a, axis=1), [np.int64, np.float64])
reg("argmin_axis1", lambda a: ak.argmin(a, axis=1), [np.int64, np.float64])
reg(
    "count_nonzero_axis1", lambda a: ak.count_nonzero(a, axis=1), [np.int64, np.float64]
)
reg("any_axis1", lambda a: ak.any(a, axis=1), [bool])
reg("all_axis1", lambda a: ak.all(a, axis=1), [bool])
reg("sum_bool_axis1", lambda a: ak.sum(a, axis=1), [bool])
reg("sum_complex_axis1", lambda a: ak.sum(a, axis=1), [np.complex128])
reg("sort_axis1", lambda a: ak.sort(a, axis=1), [np.int64, np.float64])
reg("argsort_axis1", lambda a: ak.argsort(a, axis=1), [np.int64, np.float64])


# --- cold subprocess: measure one op's marginal peak RSS, then exit ----------
if len(sys.argv) >= 5 and sys.argv[1] == "--mem-child":
    _scale, _op, _dtn = sys.argv[2], sys.argv[3], sys.argv[4]
    _n_lists, _avg_len = SCALES[_scale]
    _fn = next(fn for nm, fn, _ in OPS if nm == _op)
    _arr = make_ragged(_n_lists, _avg_len, _dtype_from_name(_dtn))
    gc.collect()
    _before = _ru_maxrss_bytes()  # high-water through build
    _res = _fn(_arr)  # the op, run once, cold
    _after = _ru_maxrss_bytes()  # high-water through the op
    print("MARGINAL_MB", max(0.0, (_after - _before) / 1e6))
    sys.exit(0)


def cold_peak_mb(scale, name, dtn):
    """Run the op once in a fresh interpreter; return its marginal peak RSS (MB)."""
    try:
        out = subprocess.run(
            [sys.executable, __file__, "--mem-child", scale, name, dtn],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        for line in out.stdout.splitlines():
            if line.startswith("MARGINAL_MB"):
                return float(line.split()[1])
    except Exception as exc:
        print(f"  (mem-child failed for {scale}|{name}|{dtn}: {exc})", flush=True)
    return None


def main():
    out_path = next(
        (a for a in sys.argv[1:] if not a.startswith("--")), "/tmp/bench_out.json"
    )
    do_mem = "--no-mem" not in sys.argv

    results = {}
    for scale_name, (n_lists, avg_len) in SCALES.items():
        cache = {}
        for name, fn, dtypes in OPS:
            for dt in dtypes:
                if dt not in cache:
                    cache[dt] = make_ragged(n_lists, avg_len, dt)
                arr = cache[dt]
                dtn = np.dtype(dt).name
                t = bench(fn, arr)
                m = cold_peak_mb(scale_name, name, dtn) if do_mem else None
                results[(scale_name, name, dtn)] = {"ms": t, "peak_mb": m}
                mstr = f"{m:8.2f} MB" if m is not None else f"{'n/a':>8}   "
                print(
                    f"{scale_name:12s} {name:22s} {dtn:12s} {t * 1e3:10.3f} ms  {mstr}",
                    flush=True,
                )
        del cache
        gc.collect()

    import json

    with open(out_path, "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, f, indent=2)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
