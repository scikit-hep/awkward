# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()


class _AllocRecorder(cp.cuda.MemoryHook):
    """Records every pool allocation made inside a `with` block."""

    name = "awkward-test-alloc-recorder"

    def __init__(self):
        self.sizes = []

    def malloc_postprocess(self, **kwargs):
        self.sizes.append(kwargs.get("mem_size", kwargs.get("size", 0)))


def _allocated_bytes(fn):
    """Total pool bytes allocated during one call of fn (steady state)."""
    fn()  # warm up: cuda.compute op JIT, caches, lazily-built buffers
    cp.cuda.Device().synchronize()
    recorder = _AllocRecorder()
    with recorder:
        fn()
        cp.cuda.Device().synchronize()
    return sum(recorder.sizes)


def _assert_no_leak(fn, calls=5):
    """Pool usage must not grow monotonically across repeated calls."""
    fn()
    cp.cuda.Device().synchronize()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline = pool.used_bytes()
    for _ in range(calls):
        fn()
    cp.cuda.Device().synchronize()
    pool.free_all_blocks()
    assert pool.used_bytes() <= baseline + 1_048_576  # 1 MB slack


def _ragged_complex(n_elements, avg_list=64, seed=0):
    rng = np.random.default_rng(seed)
    n_rows = max(1, n_elements // avg_list)
    counts = rng.poisson(avg_list, n_rows)
    total = int(counts.sum())
    data = (rng.standard_normal(total) + 1j * rng.standard_normal(total)).astype(
        np.complex128
    )
    return ak.to_backend(ak.unflatten(ak.Array(data), counts), "cuda")


def _ragged_float(n_elements, avg_list=64, seed=0):
    rng = np.random.default_rng(seed)
    n_rows = max(1, n_elements // avg_list)
    counts = rng.poisson(avg_list, n_rows)
    data = rng.standard_normal(int(counts.sum()))
    return ak.to_backend(ak.unflatten(ak.Array(data), counts), "cuda")


N = 1_000_000
INPUT_BYTES_COMPLEX = N * 16


def test_sum_complex_memory():
    array = _ragged_complex(N)
    fn = lambda: ak.sum(array, axis=1)  # noqa: E731
    _assert_no_leak(fn)
    # A pure segmented reduce needs only the output (plus O(rows) scratch);
    # anything on the order of the input signals an intermediate copy.
    assert _allocated_bytes(fn) < 0.5 * INPUT_BYTES_COMPLEX


def test_prod_complex_memory():
    array = _ragged_complex(N)
    fn = lambda: ak.prod(array, axis=1)  # noqa: E731
    _assert_no_leak(fn)
    assert _allocated_bytes(fn) < 0.5 * INPUT_BYTES_COMPLEX


def test_sum_bool_complex_memory():
    # KNOWN: the current cuda.compute implementation materializes an int8
    # intermediate of one byte per element (`mapped_data`) before the
    # segmented reduce. This test pins the *ceiling*: the intermediate is
    # N bytes, so total per-call allocations must stay well under one input
    # copy (16N). If the map is fused via TransformIterator, tighten the
    # bound to match the reducers above.
    array = _ragged_complex(N)
    fn = lambda: ak.any(array, axis=1)  # noqa: E731
    _assert_no_leak(fn)
    assert _allocated_bytes(fn) < 0.5 * INPUT_BYTES_COMPLEX


def test_rpad_and_clip_axis1_memory():
    array = _ragged_float(N)
    target = 128
    output_bytes = len(array) * target * 8
    fn = lambda: ak.pad_none(array, target, axis=1, clip=True)  # noqa: E731
    _assert_no_leak(fn)
    # Index generation is a single fused transform; allow the output index
    # plus the option-type wrapping, but not multiples of it.
    assert _allocated_bytes(fn) < 16 * output_bytes


def test_missing_repeat_memory():
    rows, cols = 100_000, 16
    data = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    array = ak.to_backend(ak.Array(data), "cuda")
    slicer = [0, None, cols - 1]
    output_bytes = rows * len(slicer) * 8
    fn = lambda: array[:, slicer]  # noqa: E731
    _assert_no_leak(fn)
    # The per-call allocation here is dominated by the slicing machinery, not the
    # kernels: NumpyArray._carry gathers the selected content (~2x output) and the
    # option-index/carry buffers (Index64.empty) account for the rest, so one slice
    # allocates ~13x output_bytes. The CUDA-compute index kernels themselves are a
    # minor, bounded share. This bound guards against gross regressions (e.g. a
    # re-introduced per-call leak or an accidental full-content duplication); the
    # strict per-call-leak check is _assert_no_leak above.
    assert _allocated_bytes(fn) < 16 * output_bytes
