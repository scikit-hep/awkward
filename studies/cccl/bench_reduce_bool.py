"""
Benchmarks for all awkward_* kernels exposed through cuda.compute (_compute.py).

Measures
--------
* GPU elapsed time  – CUDA events (min / mean / max / std over N timed runs)
* Device memory     – CuPy memory-pool delta (peak bytes allocated per call)
* Effective throughput – GB/s  (primary input/output bytes / mean kernel time)

Scales
------
  small  :   64 K elements,    512 segments
  medium :    8 M elements,  64 K segments
  large  :  256 M elements,   2 M segments

Kernels benchmarked (25)
------------------------
  bool / count reductions (6):
    awkward_reduce_sum_bool, awkward_reduce_prod_bool,
    awkward_reduce_sum_int32_bool_64, awkward_reduce_sum_int64_bool_64,
    awkward_reduce_countnonzero, awkward_reduce_count_64

  int64 reductions (6):
    awkward_reduce_sum, awkward_reduce_prod,
    awkward_reduce_max, awkward_reduce_min,
    awkward_reduce_argmax, awkward_reduce_argmin

  complex float32 reductions (9):
    awkward_reduce_sum_complex, awkward_reduce_prod_complex,
    awkward_reduce_max_complex, awkward_reduce_min_complex,
    awkward_reduce_sum_bool_complex, awkward_reduce_prod_bool_complex,
    awkward_reduce_argmax_complex, awkward_reduce_argmin_complex,
    awkward_reduce_countnonzero_complex

  index / transform (4):
    awkward_index_rpad_and_clip_axis0, awkward_index_rpad_and_clip_axis1,
    awkward_missing_repeat, segmented_sort

Usage
-----
    python bench_reduce_bool.py [--warmup 5] [--runs 30] [--nsys]
    python bench_reduce_bool.py --kernel-file src/awkward/_connect/cuda/_compute.py
    python bench_reduce_bool.py --scales small medium   # subset of scales
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# CUDA guard
# ---------------------------------------------------------------------------
try:
    import cupy as cp
except ImportError:
    sys.exit("cupy is not installed – benchmarks require a CUDA GPU.")

try:
    cp.cuda.Device(0).use()
except cp.cuda.runtime.CUDARuntimeError:
    sys.exit("No CUDA device found.")

import numpy as np

# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------
ALL_KERNEL_NAMES = [
    # bool / count
    "awkward_reduce_sum_bool",
    "awkward_reduce_prod_bool",
    "awkward_reduce_sum_int32_bool_64",
    "awkward_reduce_sum_int64_bool_64",
    "awkward_reduce_countnonzero",
    "awkward_reduce_count_64",
    # int64
    "awkward_reduce_sum",
    "awkward_reduce_prod",
    "awkward_reduce_max",
    "awkward_reduce_min",
    "awkward_reduce_argmax",
    "awkward_reduce_argmin",
    # complex float32
    "awkward_reduce_sum_complex",
    "awkward_reduce_prod_complex",
    "awkward_reduce_max_complex",
    "awkward_reduce_min_complex",
    "awkward_reduce_sum_bool_complex",
    "awkward_reduce_prod_bool_complex",
    "awkward_reduce_argmax_complex",
    "awkward_reduce_argmin_complex",
    "awkward_reduce_countnonzero_complex",
    # index / transform
    "awkward_index_rpad_and_clip_axis0",
    "awkward_index_rpad_and_clip_axis1",
    "awkward_missing_repeat",
    "segmented_sort",
]

# Filled at import time by _load_kernels()
K: dict[str, Callable] = {}


def _load_kernels(path: str | None) -> None:
    """
    Populate K with every kernel found in `path` (file) or via package import.
    Kernels missing from the module are silently excluded from K.
    """
    candidates = [
        "awkward._connect.cuda._compute",
        "awkward._backends.cuda_kernels",
        "cuda_kernels",
    ]

    mod = None
    if path:
        spec = importlib.util.spec_from_file_location("_cuda_kernels", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
    else:
        for name in candidates:
            try:
                mod = importlib.import_module(name)
                break
            except ImportError:
                continue

    if mod is None:
        sys.exit(
            "\nCould not locate the kernel module.\n"
            "Pass the path explicitly:\n\n"
            "    python bench_reduce_bool.py "
            "--kernel-file src/awkward/_connect/cuda/_compute.py\n"
        )

    missing = []
    for name in ALL_KERNEL_NAMES:
        fn = getattr(mod, name, None)
        if fn is not None:
            K[name] = fn
        else:
            missing.append(name)

    if missing:
        print(
            f"[warn] {len(missing)} kernel(s) not found in module "
            f"(will be skipped): {missing}",
            file=sys.stderr,
        )


# Pre-parse --kernel-file so the import happens before anything else
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--kernel-file", default=None)
_pre_args, _ = _pre.parse_known_args()
_load_kernels(_pre_args.kernel_file)


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------
SCALES = {
    #          n_elements    n_segments  sparsity
    "small":  (64_000,        512,       0.5),
    "medium": (8_000_000,     64_000,    0.5),
    "large":  (256_000_000,   2_000_000, 0.5),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    name:       str
    n_elements: int   # also == n_complex for the complex kernels
    n_segments: int

    # Segmented-reduction inputs
    input_bool:       "cp.ndarray"  # bool,    (n_elements,)
    input_int64:      "cp.ndarray"  # int64,   (n_elements,)
    # Interleaved complex64 stored as float32: shape (2*n_elements,).
    # offsets index the complex (not float) dimension, so offset[i] in [0..n_elements].
    input_float32_cx: "cp.ndarray"  # float32, (2*n_elements,)

    offsets: "cp.ndarray"  # int64, (n_segments+1,)

    # Output buffers
    result_bool:       "cp.ndarray"  # bool,    (n_segments,)
    result_int32:      "cp.ndarray"  # int32,   (n_segments,)
    result_int64:      "cp.ndarray"  # int64,   (n_segments,)
    # Interleaved complex64 output: shape (2*n_segments,)
    result_float32_cx: "cp.ndarray"  # float32, (2*n_segments,)

    # segmented_sort
    sort_out: "cp.ndarray"  # float32, (n_elements,)

    # awkward_index_rpad_and_clip_axis0
    rpad0_out:    "cp.ndarray"  # int64, (n_elements,)  ← target == n_elements
    rpad0_target: int           # == n_elements
    rpad0_length: int           # == n_elements // 2   (shorter = min(target, length))

    # awkward_index_rpad_and_clip_axis1
    rpad1_starts: "cp.ndarray"  # int64, (n_segments,)
    rpad1_stops:  "cp.ndarray"  # int64, (n_segments,)
    rpad1_target: int           # average segment length
    rpad1_length: int           # == n_segments

    # awkward_missing_repeat
    mr_index:  "cp.ndarray"  # int64, (n_segments,)  ~20% are -1
    mr_out:    "cp.ndarray"  # int64, (n_segments * mr_reps,) ≈ (n_elements,)
    mr_reps:   int
    mr_regsize: int


def make_dataset(
    name: str, n_elements: int, n_segments: int, sparsity: float
) -> Dataset:
    rng = np.random.default_rng(seed=42)

    # Bool
    input_bool = cp.asarray(rng.random(n_elements) < sparsity, dtype=cp.bool_)

    # Int64 – small positive values to prevent overflow in prod benchmarks
    input_int64 = cp.asarray(
        rng.integers(1, 10, size=n_elements, dtype=np.int64)
    )

    # Complex float32 (interleaved real/imag).
    # Generate in two passes to cap peak host RAM at ~2 GB for large scale.
    host_cx = np.empty(2 * n_elements, dtype=np.float32)
    host_cx[0::2] = rng.random(n_elements)
    host_cx[1::2] = rng.random(n_elements)
    input_float32_cx = cp.asarray(host_cx)
    del host_cx

    # Offsets: random segment lengths ≥ 1 summing to n_elements
    extra = n_elements - n_segments
    cuts  = np.sort(rng.choice(extra + n_segments - 1, n_segments - 1, replace=False))
    lengths = np.diff(
        np.concatenate(([0], cuts + 1, [extra + n_segments]))
    ).astype(np.int64)
    assert lengths.sum() == n_elements
    assert (lengths > 0).all()
    offsets_np = np.zeros(n_segments + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets_np[1:])
    offsets = cp.asarray(offsets_np)

    # Output buffers
    result_bool       = cp.empty(n_segments,     dtype=cp.bool_)
    result_int32      = cp.empty(n_segments,     dtype=cp.int32)
    result_int64      = cp.empty(n_segments,     dtype=cp.int64)
    result_float32_cx = cp.empty(2 * n_segments, dtype=cp.float32)
    sort_out          = cp.empty(n_elements,     dtype=cp.float32)

    # rpad_axis0: fill toindex[0..n_elements) where first half maps identity,
    # rest maps to -1.
    rpad0_out    = cp.empty(n_elements, dtype=cp.int64)
    rpad0_target = n_elements
    rpad0_length = n_elements // 2

    # rpad_axis1: one entry per segment
    rpad1_starts = cp.empty(n_segments, dtype=cp.int64)
    rpad1_stops  = cp.empty(n_segments, dtype=cp.int64)
    rpad1_target = max(1, n_elements // n_segments)
    rpad1_length = n_segments

    # missing_repeat: index of length n_segments, ~20% missing (-1)
    mr_reps    = max(1, n_elements // n_segments)
    mr_regsize = mr_reps
    host_mr    = np.arange(n_segments, dtype=np.int64)
    miss_idx   = rng.choice(n_segments, max(1, n_segments // 5), replace=False)
    host_mr[miss_idx] = -1
    mr_index = cp.asarray(host_mr)
    mr_out   = cp.empty(n_segments * mr_reps, dtype=cp.int64)

    return Dataset(
        name=name, n_elements=n_elements, n_segments=n_segments,
        input_bool=input_bool, input_int64=input_int64,
        input_float32_cx=input_float32_cx, offsets=offsets,
        result_bool=result_bool, result_int32=result_int32,
        result_int64=result_int64, result_float32_cx=result_float32_cx,
        sort_out=sort_out,
        rpad0_out=rpad0_out, rpad0_target=rpad0_target, rpad0_length=rpad0_length,
        rpad1_starts=rpad1_starts, rpad1_stops=rpad1_stops,
        rpad1_target=rpad1_target, rpad1_length=rpad1_length,
        mr_index=mr_index, mr_out=mr_out, mr_reps=mr_reps, mr_regsize=mr_regsize,
    )


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------
@dataclass
class TimingResult:
    kernel:      str
    scale:       str
    n_elements:  int
    n_segments:  int
    input_bytes: int   # bytes of primary data read/written (for GB/s)
    warmup_runs: int
    timed_runs:  int
    times_ms:    list[float] = field(default_factory=list)
    mem_delta_bytes: int = 0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)

    @property
    def std_ms(self) -> float:
        m = self.mean_ms
        return math.sqrt(sum((t - m) ** 2 for t in self.times_ms) / len(self.times_ms))

    @property
    def throughput_gbs(self) -> float:
        if self.mean_ms == 0:
            return float("inf")
        return (self.input_bytes / 1e9) / (self.mean_ms / 1e3)


def _cuda_time_call(fn: Callable, n_runs: int) -> list[float]:
    start = cp.cuda.Event()
    end   = cp.cuda.Event()
    times = []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))
    return times


def _mem_delta(fn: Callable) -> int:
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    before = pool.used_bytes()
    fn()
    cp.cuda.Device().synchronize()
    return max(0, pool.used_bytes() - before)


def benchmark_kernel(
    kernel_name: str,
    call_fn: Callable,
    input_bytes: int,
    ds: Dataset,
    warmup: int,
    runs: int,
    use_nsys_profile: bool,
) -> TimingResult:
    result = TimingResult(
        kernel=kernel_name, scale=ds.name,
        n_elements=ds.n_elements, n_segments=ds.n_segments,
        input_bytes=input_bytes, warmup_runs=warmup, timed_runs=runs,
    )
    for _ in range(warmup):
        call_fn()
    cp.cuda.Device().synchronize()

    result.mem_delta_bytes = _mem_delta(call_fn)

    if use_nsys_profile:
        with cp.cuda.profile():
            result.times_ms = _cuda_time_call(call_fn, runs)
    else:
        result.times_ms = _cuda_time_call(call_fn, runs)

    return result


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------
_KW = 42   # kernel name column width

HEADER = (
    f"{'Kernel':<{_KW}}"
    f"{'Scale':<8}"
    f"{'Elements':>13}"
    f"{'Segments':>10}"
    f"{'Min ms':>9}"
    f"{'Mean ms':>9}"
    f"{'Max ms':>9}"
    f"{'Std ms':>8}"
    f"{'GB/s':>9}"
    f"{'MemΔ KB':>9}"
)
_LINE = "─" * len(HEADER)
_DLINE = "═" * len(HEADER)


def _group_header(label: str) -> str:
    inner = f"  {label}  "
    pad   = (_LINE.__len__() - len(inner)) // 2
    return "─" * pad + inner + "─" * (len(_LINE) - pad - len(inner))


def print_result(r: TimingResult) -> None:
    mem_kb = f"{r.mem_delta_bytes / 1024:.1f}" if r.mem_delta_bytes else "0"
    print(
        f"{r.kernel:<{_KW}}"
        f"{r.scale:<8}"
        f"{r.n_elements:>13,}"
        f"{r.n_segments:>10,}"
        f"{r.min_ms:>9.3f}"
        f"{r.mean_ms:>9.3f}"
        f"{r.max_ms:>9.3f}"
        f"{r.std_ms:>8.3f}"
        f"{r.throughput_gbs:>9.2f}"
        f"{mem_kb:>9}"
    )


# ---------------------------------------------------------------------------
# Kernel call factories
# Each returns (zero-arg callable, input_bytes: int).
# input_bytes = primary bytes of data read/written, used for GB/s.
# ---------------------------------------------------------------------------

# ── bool / count reductions ────────────────────────────────────────────────

def make_sum_bool_call(ds: Dataset):
    fn = K["awkward_reduce_sum_bool"]
    def _c(): fn(ds.result_bool, ds.input_bool, ds.offsets, ds.n_segments)
    return _c, ds.n_elements                        # 1 byte/element (bool)

def make_prod_bool_call(ds: Dataset):
    fn = K["awkward_reduce_prod_bool"]
    def _c(): fn(ds.result_bool, ds.input_bool, ds.offsets, ds.n_segments)
    return _c, ds.n_elements

def make_sum_int32_bool_call(ds: Dataset):
    fn = K["awkward_reduce_sum_int32_bool_64"]
    def _c(): fn(ds.result_int32, ds.input_bool, ds.offsets, ds.n_segments)
    return _c, ds.n_elements

def make_sum_int64_bool_call(ds: Dataset):
    fn = K["awkward_reduce_sum_int64_bool_64"]
    def _c(): fn(ds.result_int64, ds.input_bool, ds.offsets, ds.n_segments)
    return _c, ds.n_elements

def make_countnonzero_call(ds: Dataset):
    fn = K["awkward_reduce_countnonzero"]
    def _c(): fn(ds.result_int64, ds.input_bool, ds.offsets, ds.n_segments)
    return _c, ds.n_elements

def make_count_64_call(ds: Dataset):
    fn = K["awkward_reduce_count_64"]
    def _c(): fn(ds.result_int64, ds.offsets, ds.n_segments)
    return _c, (ds.n_segments + 1) * 8             # reads offsets only

# ── int64 reductions ───────────────────────────────────────────────────────

def make_sum_call(ds: Dataset):
    fn = K["awkward_reduce_sum"]
    def _c(): fn(ds.result_int64, ds.input_int64, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

def make_prod_call(ds: Dataset):
    fn = K["awkward_reduce_prod"]
    def _c(): fn(ds.result_int64, ds.input_int64, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

def make_max_call(ds: Dataset):
    fn  = K["awkward_reduce_max"]
    idn = int(np.iinfo(np.int64).min)
    def _c(): fn(ds.result_int64, ds.input_int64, ds.offsets, ds.n_segments, idn)
    return _c, ds.n_elements * 8

def make_min_call(ds: Dataset):
    fn  = K["awkward_reduce_min"]
    idn = int(np.iinfo(np.int64).max)
    def _c(): fn(ds.result_int64, ds.input_int64, ds.offsets, ds.n_segments, idn)
    return _c, ds.n_elements * 8

def make_argmax_call(ds: Dataset):
    fn     = K["awkward_reduce_argmax"]
    starts = ds.offsets[:-1]               # accepted but unused by the kernel
    def _c(): fn(ds.result_int64, ds.input_int64, ds.offsets, starts, ds.n_segments)
    return _c, ds.n_elements * 8

def make_argmin_call(ds: Dataset):
    fn     = K["awkward_reduce_argmin"]
    starts = ds.offsets[:-1]
    def _c(): fn(ds.result_int64, ds.input_int64, ds.offsets, starts, ds.n_segments)
    return _c, ds.n_elements * 8

# ── complex float32 reductions ─────────────────────────────────────────────
# input_float32_cx is (2*n_elements,) float32 viewed as (n_elements,) complex64.
# offsets index the complex dimension → offset values in [0..n_elements].

def make_sum_complex_call(ds: Dataset):
    fn = K["awkward_reduce_sum_complex"]
    def _c(): fn(ds.result_float32_cx, ds.input_float32_cx, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8           # n_elements complex64 @ 8 B each

def make_prod_complex_call(ds: Dataset):
    fn = K["awkward_reduce_prod_complex"]
    def _c(): fn(ds.result_float32_cx, ds.input_float32_cx, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

def make_max_complex_call(ds: Dataset):
    fn  = K["awkward_reduce_max_complex"]
    idn = float("-inf")
    def _c(): fn(ds.result_float32_cx, ds.input_float32_cx, ds.offsets, ds.n_segments, idn)
    return _c, ds.n_elements * 8

def make_min_complex_call(ds: Dataset):
    fn  = K["awkward_reduce_min_complex"]
    idn = float("inf")
    def _c(): fn(ds.result_float32_cx, ds.input_float32_cx, ds.offsets, ds.n_segments, idn)
    return _c, ds.n_elements * 8

def make_sum_bool_complex_call(ds: Dataset):
    fn = K["awkward_reduce_sum_bool_complex"]
    def _c(): fn(ds.result_bool, ds.input_float32_cx, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

def make_prod_bool_complex_call(ds: Dataset):
    fn = K["awkward_reduce_prod_bool_complex"]
    def _c(): fn(ds.result_bool, ds.input_float32_cx, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

def make_argmax_complex_call(ds: Dataset):
    fn = K["awkward_reduce_argmax_complex"]
    def _c(): fn(ds.result_int64, ds.input_float32_cx, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

def make_argmin_complex_call(ds: Dataset):
    fn = K["awkward_reduce_argmin_complex"]
    def _c(): fn(ds.result_int64, ds.input_float32_cx, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

def make_countnonzero_complex_call(ds: Dataset):
    fn = K["awkward_reduce_countnonzero_complex"]
    def _c(): fn(ds.result_int64, ds.input_float32_cx, ds.offsets, ds.n_segments)
    return _c, ds.n_elements * 8

# ── index / transform ──────────────────────────────────────────────────────

def make_rpad0_call(ds: Dataset):
    fn = K["awkward_index_rpad_and_clip_axis0"]
    def _c(): fn(ds.rpad0_out, ds.rpad0_target, ds.rpad0_length)
    return _c, ds.rpad0_target * 8         # writes n_elements int64

def make_rpad1_call(ds: Dataset):
    fn = K["awkward_index_rpad_and_clip_axis1"]
    def _c(): fn(ds.rpad1_starts, ds.rpad1_stops, ds.rpad1_target, ds.rpad1_length)
    return _c, ds.n_segments * 8 * 2       # writes 2 × n_segments int64

def make_missing_repeat_call(ds: Dataset):
    fn = K["awkward_missing_repeat"]
    def _c():
        fn(ds.mr_out, ds.mr_index, ds.n_segments, ds.mr_reps, ds.mr_regsize)
    return _c, ds.n_segments * ds.mr_reps * 8   # writes n_elements int64

def make_sort_call(ds: Dataset):
    fn  = K["segmented_sort"]
    src = ds.input_float32_cx[: ds.n_elements]   # first n_elements floats
    nel = ds.n_elements
    nof = ds.n_segments + 1
    def _c(): fn(ds.sort_out, src, nel, ds.offsets, nof, nel, True, True)
    return _c, nel * 4                      # reads n_elements float32


# ---------------------------------------------------------------------------
# Kernel groups (order controls display order)
# ---------------------------------------------------------------------------
KERNEL_GROUPS: list[tuple[str, list[tuple[str, Callable]]]] = [
    ("bool / count reductions", [
        ("awkward_reduce_sum_bool",          make_sum_bool_call),
        ("awkward_reduce_prod_bool",         make_prod_bool_call),
        ("awkward_reduce_sum_int32_bool_64", make_sum_int32_bool_call),
        ("awkward_reduce_sum_int64_bool_64", make_sum_int64_bool_call),
        ("awkward_reduce_countnonzero",      make_countnonzero_call),
        ("awkward_reduce_count_64",          make_count_64_call),
    ]),
    ("int64 reductions", [
        ("awkward_reduce_sum",    make_sum_call),
        ("awkward_reduce_prod",   make_prod_call),
        ("awkward_reduce_max",    make_max_call),
        ("awkward_reduce_min",    make_min_call),
        ("awkward_reduce_argmax", make_argmax_call),
        ("awkward_reduce_argmin", make_argmin_call),
    ]),
    ("complex float32 reductions", [
        ("awkward_reduce_sum_complex",          make_sum_complex_call),
        ("awkward_reduce_prod_complex",         make_prod_complex_call),
        ("awkward_reduce_max_complex",          make_max_complex_call),
        ("awkward_reduce_min_complex",          make_min_complex_call),
        ("awkward_reduce_sum_bool_complex",     make_sum_bool_complex_call),
        ("awkward_reduce_prod_bool_complex",    make_prod_bool_complex_call),
        ("awkward_reduce_argmax_complex",       make_argmax_complex_call),
        ("awkward_reduce_argmin_complex",       make_argmin_complex_call),
        ("awkward_reduce_countnonzero_complex", make_countnonzero_complex_call),
    ]),
    ("index / transform", [
        ("awkward_index_rpad_and_clip_axis0", make_rpad0_call),
        ("awkward_index_rpad_and_clip_axis1", make_rpad1_call),
        ("awkward_missing_repeat",            make_missing_repeat_call),
        ("segmented_sort",                    make_sort_call),
    ]),
]


# ---------------------------------------------------------------------------
# Correctness smoke tests
# ---------------------------------------------------------------------------
def smoke_test_bool_reductions() -> None:
    # 3 segments: [T F T] | [] | [F F F T]
    inp = cp.asarray([True, False, True, False, False, False, True], dtype=cp.bool_)
    off = cp.asarray([0, 3, 3, 7], dtype=cp.int64)
    n   = 3

    res = cp.empty(n, dtype=cp.bool_)
    K["awkward_reduce_prod_bool"](res, inp, off, n)
    assert cp.asnumpy(res).tolist() == [False, True, False], \
        f"prod_bool FAIL: {cp.asnumpy(res)}"

    K["awkward_reduce_sum_bool"](res, inp, off, n)
    assert cp.asnumpy(res).tolist() == [True, False, True], \
        f"sum_bool FAIL: {cp.asnumpy(res)}"

    res64 = cp.empty(n, dtype=cp.int64)
    K["awkward_reduce_countnonzero"](res64, inp, off, n)
    assert cp.asnumpy(res64).tolist() == [2, 0, 1], \
        f"countnonzero FAIL: {cp.asnumpy(res64)}"

    K["awkward_reduce_count_64"](res64, off, n)
    assert cp.asnumpy(res64).tolist() == [3, 0, 4], \
        f"count_64 FAIL: {cp.asnumpy(res64)}"


def smoke_test_int64_reductions() -> None:
    # 3 segments: [1,2,3] | [4,5] | [6]
    inp = cp.asarray([1, 2, 3, 4, 5, 6], dtype=cp.int64)
    off = cp.asarray([0, 3, 5, 6], dtype=cp.int64)
    n   = 3

    res = cp.empty(n, dtype=cp.int64)
    K["awkward_reduce_sum"](res, inp, off, n)
    assert cp.asnumpy(res).tolist() == [6, 9, 6], \
        f"sum FAIL: {cp.asnumpy(res)}"

    K["awkward_reduce_max"](res, inp, off, n, int(np.iinfo(np.int64).min))
    assert cp.asnumpy(res).tolist() == [3, 5, 6], \
        f"max FAIL: {cp.asnumpy(res)}"

    K["awkward_reduce_min"](res, inp, off, n, int(np.iinfo(np.int64).max))
    assert cp.asnumpy(res).tolist() == [1, 4, 6], \
        f"min FAIL: {cp.asnumpy(res)}"

    starts = off[:-1]
    K["awkward_reduce_argmax"](res, inp, off, starts, n)
    # argmax returns global index: seg0→idx2(val3), seg1→idx4(val5), seg2→idx5(val6)
    assert cp.asnumpy(res).tolist() == [2, 4, 5], \
        f"argmax FAIL: {cp.asnumpy(res)}"

    K["awkward_reduce_argmin"](res, inp, off, starts, n)
    assert cp.asnumpy(res).tolist() == [0, 3, 5], \
        f"argmin FAIL: {cp.asnumpy(res)}"


def smoke_test_rpad() -> None:
    # rpad_axis0: target=6, length=4 → [0,1,2,3,-1,-1]
    out = cp.empty(6, dtype=cp.int64)
    K["awkward_index_rpad_and_clip_axis0"](out, 6, 4)
    assert cp.asnumpy(out).tolist() == [0, 1, 2, 3, -1, -1], \
        f"rpad_axis0 FAIL: {cp.asnumpy(out)}"

    # rpad_axis1: 3 lists, target=2 → starts=[0,2,4], stops=[2,4,6]
    starts = cp.empty(3, dtype=cp.int64)
    stops  = cp.empty(3, dtype=cp.int64)
    K["awkward_index_rpad_and_clip_axis1"](starts, stops, 2, 3)
    assert cp.asnumpy(starts).tolist() == [0, 2, 4], \
        f"rpad_axis1 starts FAIL: {cp.asnumpy(starts)}"
    assert cp.asnumpy(stops).tolist() == [2, 4, 6], \
        f"rpad_axis1 stops FAIL: {cp.asnumpy(stops)}"


def run_smoke_tests() -> None:
    smoke_test_bool_reductions()
    smoke_test_int64_reductions()
    smoke_test_rpad()
    print("  smoke tests passed ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all awkward_* cuda.compute kernels"
    )
    parser.add_argument("--kernel-file", default=None, metavar="PATH",
                        help="Path to the kernel module .py file")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations per kernel (default: 5)")
    parser.add_argument("--runs", type=int, default=30,
                        help="Timed iterations per kernel×scale (default: 30)")
    parser.add_argument("--nsys", action="store_true",
                        help="Wrap timed loops in cupy.cuda.profile() for Nsight Systems")
    parser.add_argument("--scales", nargs="+", choices=list(SCALES),
                        default=list(SCALES),
                        help="Scales to run (default: all)")
    parser.add_argument("--groups", nargs="+",
                        choices=[g for g, _ in KERNEL_GROUPS],
                        default=None,
                        help="Kernel groups to run (default: all)")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip correctness smoke tests")
    args = parser.parse_args()

    active_groups = args.groups or [g for g, _ in KERNEL_GROUPS]

    # Header
    props    = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(f"\n{_DLINE}")
    print(f"  GPU : {gpu_name}")
    print(f"  CUDA driver : {cp.cuda.runtime.driverGetVersion()}")
    print(f"  CuPy        : {cp.__version__}")
    print(f"  Warmup runs : {args.warmup}   Timed runs: {args.runs}")
    print(f"  Scales      : {', '.join(args.scales)}")
    print(f"{_DLINE}\n")

    # Smoke tests
    if not args.skip_smoke:
        print("Running correctness smoke tests …")
        run_smoke_tests()
        print()

    # Build datasets
    datasets: dict[str, Dataset] = {}
    for scale_name in args.scales:
        n_el, n_seg, sparsity = SCALES[scale_name]
        print(
            f"Allocating {scale_name} dataset "
            f"({n_el:,} elements, {n_seg:,} segments) …",
            end=" ", flush=True,
        )
        t0 = time.perf_counter()
        datasets[scale_name] = make_dataset(scale_name, n_el, n_seg, sparsity)
        cp.cuda.Device().synchronize()
        print(f"done in {time.perf_counter() - t0:.2f}s")
    print()

    # Run
    all_results: list[TimingResult] = []

    print(HEADER)

    for group_label, kernels in KERNEL_GROUPS:
        if group_label not in active_groups:
            continue

        print(_group_header(group_label))

        for kernel_name, factory_fn in kernels:
            if kernel_name not in K:
                print(f"  [skip] {kernel_name} – not found in module")
                continue

            for scale_name in args.scales:
                ds = datasets[scale_name]
                try:
                    call_fn, input_bytes = factory_fn(ds)
                except Exception as exc:
                    print(f"  [skip] {kernel_name}/{scale_name} – factory error: {exc}")
                    continue

                try:
                    r = benchmark_kernel(
                        kernel_name=kernel_name,
                        call_fn=call_fn,
                        input_bytes=input_bytes,
                        ds=ds,
                        warmup=args.warmup,
                        runs=args.runs,
                        use_nsys_profile=args.nsys,
                    )
                except Exception as exc:
                    print(f"  [error] {kernel_name}/{scale_name}: {exc}")
                    continue

                all_results.append(r)
                print_result(r)

    print(_DLINE)

    # Per-group throughput summary
    print("\nMean GB/s by group and scale:")
    for group_label, kernels in KERNEL_GROUPS:
        if group_label not in active_groups:
            continue
        knames = {kn for kn, _ in kernels}
        group_results = [r for r in all_results if r.kernel in knames]
        if not group_results:
            continue
        print(f"\n  {group_label}")
        for scale_name in args.scales:
            scale_res = [r for r in group_results if r.scale == scale_name]
            if scale_res:
                avg = sum(r.throughput_gbs for r in scale_res) / len(scale_res)
                print(f"    {scale_name:<8}  {avg:6.2f} GB/s  "
                      f"(across {len(scale_res)} kernel(s))")

    print()
    print("Note: MemΔ shows pool-delta after JIT warmup.  Kernels that allocate")
    print("      an intermediate array (mapped, mapped_data) show non-zero MemΔ;")
    print("      direct-write kernels (reduce_sum, reduce_max, sort, …) show 0.")
    print()


if __name__ == "__main__":
    main()

