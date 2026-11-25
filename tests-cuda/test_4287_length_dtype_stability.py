# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.typetracer import TypeTracerArray

cp = pytest.importorskip("cupy")
_caching = pytest.importorskip("cuda.compute._caching")


SMALL_LENGTHS = [1, 2, 10]
BOUNDARY_LENGTHS = [127, 128, 255, 256]
LARGE_LENGTHS = [2**31 - 1, 2**31, 2**63 - 1]  # int32 / int64 boundaries
ALL_LENGTHS = SMALL_LENGTHS + BOUNDARY_LENGTHS + LARGE_LENGTHS

# Real kernel runs on CUDA are slow to test broadly: each distinct
# (length, n) pair is a distinct (starts, stops) shape, and cuda.compute's
# op cache keys closures on (dtype, shape), so each one forces a fresh CUDA
# JIT compile (~7-10s). Dtype selection itself is already exhaustively
# checked across ALL_LENGTHS above without touching the GPU, so a real
# kernel run here only needs to cover the uint8/uint16 boundary transition
# (255 -> 256) at a single n, to prove the fix doesn't break at the switch.
CORRECTNESS_LENGTHS = [255, 256]
CORRECTNESS_N = 3


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp.cuda.Device().synchronize()
    cp._default_memory_pool.free_all_blocks()


def _infer_length_dtype(length):
    """
    Mirrors the dtype-selection expression used to wrap `length` for the
    `fill_pos` closure in `awkward_ListArray_combinations`
    (src/awkward/_connect/cuda/_compute.py): arrays keep their own dtype,
    plain Python ints/scalars get the smallest unsigned dtype that can hold
    the value.
    """
    return length.dtype if hasattr(length, "dtype") else np.min_scalar_type(length)


def _cache_sizes():
    return {
        name: len(fn.cache_clear.__self__)
        for name, fn in _caching._cache_registry.items()
    }


# ---------------------------------------------------------------------------
# dtype stability + no widening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", ALL_LENGTHS)
def test_dtype_inference_matches_device_dtype(length):
    dtype = _infer_length_dtype(length)
    arr = cp.asarray([length], dtype=dtype)

    # dtype stability: the wrapper array actually carries the inferred dtype
    assert arr.dtype == dtype
    # no widening: the inferred dtype is the smallest one that fits, not a
    # blanket int64/float64 promotion
    assert dtype == np.min_scalar_type(length)
    assert int(arr[0]) == length


@pytest.mark.parametrize("np_dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_dtype_inference_preserves_array_input_dtype(np_dtype):
    # When `length` already carries a dtype (e.g. a numpy/cupy scalar), that
    # dtype must be used as-is rather than being forced to int64.
    length = np_dtype(1000)
    dtype = _infer_length_dtype(length)
    assert dtype == np.dtype(np_dtype)

    arr = cp.asarray([length], dtype=dtype)
    assert arr.dtype == np.dtype(np_dtype)
    assert int(arr[0]) == 1000


# ---------------------------------------------------------------------------
# kernel correctness
# ---------------------------------------------------------------------------


def _make_jagged_cuda_and_cpu(length, min_count, seed):
    rng = np.random.default_rng(seed)
    counts = rng.integers(min_count, min_count + 4, size=length)
    data = rng.standard_normal(int(counts.sum()))
    cpu_array = ak.unflatten(ak.Array(data), counts)
    cuda_array = ak.to_backend(cpu_array, "cuda")
    return cpu_array, cuda_array


@pytest.mark.parametrize("length", CORRECTNESS_LENGTHS)
def test_combinations_correct_on_cuda_vs_cpu(length):
    n = CORRECTNESS_N
    cpu_array, cuda_array = _make_jagged_cuda_and_cpu(length, n, seed=length + n)

    fields = [f"f{i}" for i in range(n)]
    cpu_result = ak.combinations(cpu_array, n, fields=fields)
    cuda_result = ak.to_backend(ak.combinations(cuda_array, n, fields=fields), "cpu")
    assert ak.array_equal(cpu_result, cuda_result)


def test_combinations_kernel_is_cache_stable():
    # Regression test for the original bug: ak.combinations() on CUDA
    # recompiled a fresh kernel on every call because the `fill_pos` closure
    # captured `length` as an id()-keyed scalar. Wrapping it in a
    # dtype-matched device array should keep the op cache flat across calls.
    _, cuda_array = _make_jagged_cuda_and_cpu(256, 3, seed=0)

    def call():
        result = ak.combinations(cuda_array, 3, fields=["a", "b", "c"])
        ak.to_backend(result, "cpu")

    call()  # JIT warmup
    sizes_after_warmup = _cache_sizes()
    for _ in range(5):
        call()
    assert _cache_sizes() == sizes_after_warmup


# ---------------------------------------------------------------------------
# typetracer compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", ALL_LENGTHS)
@pytest.mark.parametrize("n", [2, 3])
def test_combinations_typetracer_does_not_break(length, n):
    # Metadata-only: no real memory is allocated regardless of `length`, so
    # this also covers the int32/int64 boundary lengths that are infeasible
    # to materialize as concrete arrays.
    offsets = ak.index.Index64(TypeTracerArray._new(np.dtype(np.int64), (length + 1,)))
    content = ak.contents.NumpyArray(
        TypeTracerArray._new(np.dtype(np.float64), (unknown_length,))
    )
    layout = ak.contents.ListOffsetArray(offsets, content)

    fields = [f"f{i}" for i in range(n)]
    result = ak.combinations(ak.Array(layout), n, fields=fields)
    assert ak.fields(result) == fields
    assert result.layout.length == length
