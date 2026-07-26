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
    out_gpu = ak.to_list(ak.to_backend(ak.var(gpu, axis=-1), "cpu"))
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
    out_gpu = ak.to_list(ak.to_backend(ak.moment(gpu, n, axis=-1), "cpu"))
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
    out_gpu = ak.to_list(ak.to_backend(ak.moment(gpu, 3, axis=-1), "cpu"))
    for a, b in zip(out_cpu, out_gpu, strict=True):
        assert b == pytest.approx(a)
