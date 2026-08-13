# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


INTEGER_DTYPES = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
]
FLOAT_DTYPES = ["float32", "float64"]


def _cpu_array(dtype):
    """A ragged array with empty lists, duplicates, and (for floats) NaNs."""
    if dtype == "bool":
        segs = [[True, False, True, True], [], [False, False], [True]]
        return ak.Array(segs)
    if dtype in FLOAT_DTYPES:
        nan = float("nan")
        inf = float("inf")
        # Mix of finite values, duplicates, NaN, +-inf, and signed zero -- all of
        # which must match the CPU ordering (NaN to the front, -0.0 == +0.0).
        segs = [
            [3.0, nan, 1.0, 1.0],
            [],
            [nan, nan, 4.0],
            [2.0],
            [9.0, 0.0, 0.0, 7.0, 7.0],
            [-inf, nan, 0.0, inf],
            [inf, -inf, -0.0, 0.0, nan, -2.5],
        ]
        return ak.values_astype(ak.Array(segs), dtype)
    # signed/unsigned integers: keep values non-negative so uint is valid
    segs = [[3, 1, 2, 1], [], [5, 5, 4], [2], [9, 0, 0, 7, 7]]
    return ak.values_astype(ak.Array(segs), dtype)


def _nan_aware_equal(a, b):
    """Compare nested lists treating NaN == NaN."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _nan_aware_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


@pytest.mark.parametrize("dtype", ["bool", *INTEGER_DTYPES, *FLOAT_DTYPES])
@pytest.mark.parametrize("ascending", [True, False])
def test_argsort_matches_cpu_stable(dtype, ascending):
    # stable=True gives a deterministic permutation, so CUDA and CPU must return
    # the exact same indices (including NaN placement at the front).
    cpu = _cpu_array(dtype)
    gpu = ak.to_backend(cpu, "cuda")

    out_cpu = ak.argsort(cpu, axis=-1, ascending=ascending, stable=True)
    out_gpu = ak.argsort(gpu, axis=-1, ascending=ascending, stable=True)

    assert ak.to_list(ak.to_backend(out_gpu, "cpu")) == ak.to_list(out_cpu)


@pytest.mark.parametrize("dtype", ["bool", *INTEGER_DTYPES, *FLOAT_DTYPES])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
def test_argsort_carry_reproduces_sort(dtype, ascending, stable):
    # Regardless of tie-breaking, gathering with the argsort result must yield
    # the same ordering as ak.sort. Compare against the CPU backend (the source
    # of truth) so this validates the GPU permutation even when stable=False,
    # where exact indices may legitimately differ.
    cpu = _cpu_array(dtype)
    gpu = ak.to_backend(cpu, "cuda")

    carry = ak.argsort(gpu, axis=-1, ascending=ascending, stable=stable)
    via_carry = ak.to_list(ak.to_backend(gpu[carry], "cpu"))
    sorted_cpu = ak.to_list(ak.sort(cpu, axis=-1, ascending=ascending, stable=stable))

    assert _nan_aware_equal(via_carry, sorted_cpu)


@pytest.mark.parametrize("dtype", ["bool", *INTEGER_DTYPES, *FLOAT_DTYPES])
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_matches_cpu(dtype, ascending):
    # Cross-check the sibling awkward_sort (segmented_sort) path too.
    cpu = _cpu_array(dtype)
    gpu = ak.to_backend(cpu, "cuda")

    out_cpu = ak.to_list(ak.sort(cpu, axis=-1, ascending=ascending, stable=True))
    out_gpu = ak.to_list(
        ak.to_backend(ak.sort(gpu, axis=-1, ascending=ascending, stable=True), "cpu")
    )

    assert _nan_aware_equal(out_gpu, out_cpu)


@pytest.mark.parametrize("ascending", [True, False])
def test_argsort_nan_with_real_infinities(ascending):
    # Regression: NaN must sort strictly before a real -inf (ascending) and stay
    # at the front for descending too -- the case a single -inf/+inf sentinel
    # could not express. e.g. [-inf, nan, 0, inf] ascending -> indices [1,0,2,3].
    cpu = ak.Array([[-float("inf"), float("nan"), 0.0, float("inf")]])
    gpu = ak.to_backend(cpu, "cuda")

    out_cpu = ak.argsort(cpu, axis=-1, ascending=ascending, stable=True)
    out_gpu = ak.argsort(gpu, axis=-1, ascending=ascending, stable=True)

    assert ak.to_list(ak.to_backend(out_gpu, "cpu")) == ak.to_list(out_cpu)


def test_argsort_all_empty():
    gpu = ak.to_backend(ak.Array([[], [], []]), "cuda")
    out = ak.to_list(ak.to_backend(ak.argsort(gpu, axis=-1), "cpu"))
    assert out == [[], [], []]


def test_segmented_argsort_non_int64_offsets():
    # Directly exercise the offsets-dtype normalization: when offsets are not
    # int64 they must be cast before segmented_sort (segmented_argsort's
    # `offsets.astype(int64)` branch).
    from awkward._connect.cuda import _compute

    fromptr = cp.asarray([3.0, 1.0, 2.0, 5.0, 4.0], dtype=cp.float64)
    offsets = cp.asarray([0, 3, 5], dtype=cp.int32)  # <- non-int64 offsets
    toptr = cp.empty(5, dtype=cp.int64)

    _compute.segmented_argsort(toptr, fromptr, 5, offsets, len(offsets), True, True)

    # segment [3,1,2] -> local [1,2,0]; segment [5,4] -> local [1,0]
    assert cp.asnumpy(toptr).tolist() == [1, 2, 0, 1, 0]


@pytest.mark.parametrize("kind_dtype", ["datetime64[D]", "timedelta64[s]"])
def test_segmented_argsort_datetime_keys(kind_dtype):
    # Exercise the datetime/timedelta branch, which views the keys as int64
    # before sorting (segmented_argsort's `keys_in.view(int64)`). cupy often
    # cannot *create/compute* datetime arrays, but a datetime64/timedelta64 view
    # of an int64 buffer (same 8-byte width) is a pure dtype reinterpret and
    # needs no datetime support -- exactly the shape the layout hands the kernel.
    from awkward._connect.cuda import _compute

    base = cp.asarray([3, 1, 2], dtype=cp.int64)
    try:
        fromptr = base.view(kind_dtype)
    except (TypeError, ValueError):
        pytest.skip(f"cupy build cannot view arrays as {kind_dtype}")
    assert fromptr.dtype.kind in "Mm"

    offsets = cp.asarray([0, 3], dtype=cp.int64)
    toptr = cp.empty(3, dtype=cp.int64)

    _compute.segmented_argsort(toptr, fromptr, 3, offsets, len(offsets), True, True)

    # payloads 3, 1, 2 ascending -> local indices [1, 2, 0]
    assert cp.asnumpy(toptr).tolist() == [1, 2, 0]
