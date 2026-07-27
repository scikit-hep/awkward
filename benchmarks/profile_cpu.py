"""CPU reduction/sort benchmark targeting the parents->offsets kernel migration."""

from __future__ import annotations

import gc
import json
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
    # warmup
    fn(arr)
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

results = {}
for scale_name, (n_lists, avg_len) in SCALES.items():
    cache = {}
    for name, fn, dtypes in OPS:
        for dt in dtypes:
            key = id(dt) if not isinstance(dt, type) else dt
            if dt not in cache:
                cache[dt] = make_ragged(n_lists, avg_len, dt)
            arr = cache[dt]
            dtn = np.dtype(dt).name
            t = bench(fn, arr)
            results[(scale_name, name, dtn)] = t
            print(
                f"{scale_name:12s} {name:22s} {dtn:12s} {t * 1e3:10.3f} ms", flush=True
            )
    del cache
    gc.collect()

with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_out.json", "w") as f:
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, f, indent=2)
print("WROTE", sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_out.json")
