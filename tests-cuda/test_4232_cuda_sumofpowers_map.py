# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")

# Coverage for the cuda.compute map ops in _connect/cuda/_compute.py that widen
# each element to float64 on the fly inside a TransformIterator feeding
# segmented_reduce (no materialised x*x / x**n buffer, no integer/float32
# overflow):
#   * _make_square_to_float64._sq   -- v = float(x); return v * v   (ak.var/std)
#   * _make_power_to_float64._pow   -- return float(x) ** n         (ak.moment)
#
# The widening happens inside the device map op, and the segmented reduction runs
# via cuda.compute, so the whole computation stays on the GPU. `_gpu` below asserts
# device residency (the op did not silently fall back to / round-trip through the
# host) before values are copied to the CPU for comparison.


def _gpu(array):
    """Assert an array is still on the cuda backend, then return it."""
    assert ak.backend(array) == "cuda"
    return array


@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "int16", "int32", "int64", "float32", "float64"]
)
def test_var_square_map_matches_cpu(dtype):
    # ak.var routes through the square map op; check every supported dtype so the
    # per-dtype cached _sq map is built and exercised.
    segs = [[3, 1, 2, 4], [], [40, 50], [7]]
    cpu = ak.values_astype(ak.Array(segs), dtype)
    gpu = ak.to_backend(cpu, "cuda")

    out_cpu = ak.to_list(ak.var(cpu, axis=-1))
    out_gpu = ak.to_list(ak.to_backend(_gpu(ak.var(gpu, axis=-1)), "cpu"))
    for a, b in zip(out_cpu, out_gpu, strict=True):
        if a is None or (isinstance(a, float) and np.isnan(a)):
            assert b is None or np.isnan(b)
        else:
            assert b == pytest.approx(a)


def test_var_square_map_int32_overflow():
    # float(x) widening inside _sq means 100000**2 (which overflows int32) is
    # computed in double precision on the device.
    cpu = ak.values_astype(ak.Array([[100000, 200000, 300000]]), np.int32)
    gpu = ak.to_backend(cpu, "cuda")
    result = ak.to_list(ak.to_backend(ak.var(gpu, axis=-1), "cpu"))[0]
    assert result == pytest.approx(np.var([100000.0, 200000.0, 300000.0]))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_moment_power_map_matches_cpu(n):
    # ak.moment(x, n) routes through the power map op _pow (cached per (dtype, n)).
    segs = [[3, 1, 2, 4], [], [5, 6], [7]]
    cpu = ak.values_astype(ak.Array(segs), np.int64)
    gpu = ak.to_backend(cpu, "cuda")

    out_cpu = ak.to_list(ak.moment(cpu, n, axis=-1))
    out_gpu = ak.to_list(ak.to_backend(_gpu(ak.moment(gpu, n, axis=-1)), "cpu"))
    for a, b in zip(out_cpu, out_gpu, strict=True):
        if a is None or (isinstance(a, float) and np.isnan(a)):
            assert b is None or np.isnan(b)
        else:
            assert b == pytest.approx(a)


def test_moment_power_map_int32_overflow():
    # float(x) ** n widening: 3000**4 overflows int32 but is exact in float64.
    cpu = ak.values_astype(ak.Array([[1000, 2000, 3000]]), np.int32)
    gpu = ak.to_backend(cpu, "cuda")
    result = ak.to_list(ak.to_backend(ak.moment(gpu, 4, axis=-1), "cpu"))[0]
    expected = np.mean(np.array([1000.0, 2000.0, 3000.0]) ** 4)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
def test_moment_power_map_dtypes(dtype):
    # Build the _pow map for several input dtypes at a fixed power.
    segs = [[2, 4, 6], [8], [10, 12]]
    cpu = ak.values_astype(ak.Array(segs), dtype)
    gpu = ak.to_backend(cpu, "cuda")
    out_cpu = ak.to_list(ak.moment(cpu, 3, axis=-1))
    out_gpu = ak.to_list(ak.to_backend(_gpu(ak.moment(gpu, 3, axis=-1)), "cpu"))
    for a, b in zip(out_cpu, out_gpu, strict=True):
        assert b == pytest.approx(a)


# --- device residency: nothing round-trips through the host ------------------


def test_all_touched_ops_stay_on_device():
    # Every statistic touched by the PR must keep its result on the cuda backend
    # for a segmented (axis=-1) reduction -- proving the elementwise centring,
    # squaring/powering and the segmented reductions all run on the GPU.
    x = ak.to_backend(ak.values_astype(ak.Array([[3, 1, 2], [4, 5]]), np.int32), "cuda")
    y = ak.to_backend(ak.values_astype(ak.Array([[2, 4, 6], [1, 3]]), np.int32), "cuda")
    w = ak.to_backend(ak.values_astype(ak.Array([[1, 2, 1], [2, 1]]), np.int32), "cuda")

    assert ak.backend(ak.sum(x, axis=-1)) == "cuda"
    assert ak.backend(ak.mean(x, axis=-1)) == "cuda"
    assert ak.backend(ak.var(x, axis=-1)) == "cuda"
    assert ak.backend(ak.std(x, axis=-1)) == "cuda"
    assert ak.backend(ak.moment(x, 3, axis=-1)) == "cuda"
    assert ak.backend(ak.covar(x, y, axis=-1)) == "cuda"
    assert ak.backend(ak.corr(x, y, axis=-1)) == "cuda"
    # weighted paths (extra products) and complex var must also stay resident
    assert ak.backend(ak.var(x, weight=w, axis=-1)) == "cuda"
    assert ak.backend(ak.mean(x, weight=w, axis=-1)) == "cuda"
    xc = ak.to_backend(ak.Array([[1 + 2j, 3 + 4j], [5 + 1j, 2 + 0j]]), "cuda")
    assert ak.backend(ak.var(xc, axis=-1)) == "cuda"


def test_full_reduction_keeps_input_on_device():
    # For axis=None the *result* is a single scalar, but the O(n) input must never
    # be pulled to the host. Using keepdims=True the reduction returns a device
    # array (a 0-d/1-d cuda array), confirming no whole-array host round-trip.
    x = ak.to_backend(ak.values_astype(ak.Array([[3, 1, 2], [4, 5]]), np.int32), "cuda")
    for op in (ak.mean, ak.var, ak.std):
        assert ak.backend(op(x, axis=None, keepdims=True)) == "cuda"
    assert ak.backend(ak.moment(x, 3, axis=None, keepdims=True)) == "cuda"
