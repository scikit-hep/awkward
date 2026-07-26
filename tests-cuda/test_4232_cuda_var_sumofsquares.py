# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")

# ak.var/ak.std on the cuda backend accumulate sum(x**2) in float64 via a
# TransformIterator (square) feeding segmented_reduce -- no x*x buffer, no
# integer/float32 overflow. Cross-checked against the CPU backend.


@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "int16", "int32", "int64", "float32", "float64"]
)
def test_var_matches_cpu(dtype):
    segs = [[3, 1, 2, 4], [], [40, 50], [7]]
    cpu = ak.values_astype(ak.Array(segs), dtype)
    gpu = ak.to_backend(cpu, "cuda")

    for op in (ak.var, ak.std):
        out_cpu = ak.to_list(op(cpu, axis=-1))
        out_gpu = ak.to_list(ak.to_backend(op(gpu, axis=-1), "cpu"))
        for a, b in zip(out_cpu, out_gpu, strict=True):
            if a is None or (isinstance(a, float) and np.isnan(a)):
                assert b is None or np.isnan(b)
            else:
                assert b == pytest.approx(a)


def test_var_int32_overflow_matches_numpy():
    # 100000**2 overflows int32; the GPU float64 accumulation must still be right.
    cpu = ak.values_astype(ak.Array([[100000, 200000, 300000]]), np.int32)
    gpu = ak.to_backend(cpu, "cuda")
    result = ak.to_list(ak.to_backend(ak.var(gpu, axis=-1), "cpu"))[0]
    assert result == pytest.approx(np.var([100000.0, 200000.0, 300000.0]))


def test_var_axis_none_matches_cpu():
    cpu = ak.values_astype(ak.Array([[1, 2, 3], [4, 5]]), np.int32)
    gpu = ak.to_backend(cpu, "cuda")
    # axis=None returns a 0-d scalar that stays on the device (cupy). Bring it to
    # the host *explicitly* for the comparison -- cupy (correctly) refuses the
    # implicit np.asarray() that pytest.approx would otherwise trigger.
    assert float(ak.var(gpu)) == pytest.approx(float(ak.var(cpu)))


# --- the two-pass regressions, on the device -------------------------------


def test_var_float32_cancellation_gpu():
    # One-pass E[x**2]-E[x]**2 loses all precision here; two-pass (float64 mean +
    # sum-of-squares of deviations) gives 2/3 on the GPU too.
    cpu = ak.Array(np.array([1e7, 1e7 + 1, 1e7 + 2], dtype=np.float32))
    gpu = ak.to_backend(cpu, "cuda")
    assert float(ak.var(gpu)) == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_var_all_equal_not_nan_gpu():
    # Large run of identical float32 values: must be exactly 0, std not nan.
    cpu = ak.Array(np.full(1_000_000, 1000.0, dtype=np.float32))
    gpu = ak.to_backend(cpu, "cuda")
    assert float(ak.var(gpu)) == pytest.approx(0.0, abs=1e-6)
    assert not np.isnan(float(ak.std(gpu)))


def test_mean_int64_overflow_gpu():
    # float64 accumulator on the device: mean(2**62 x4) must not wrap to 0.
    cpu = ak.Array(np.array([2**62] * 4, dtype=np.int64))
    gpu = ak.to_backend(cpu, "cuda")
    assert float(ak.mean(gpu)) == pytest.approx(float(2**62))


def test_var_complex_is_real_gpu():
    # Variance of complex data is real (E[|x-mean|**2]); check the device path.
    data = np.array([1 + 2j, 3 + 4j, 5 + 1j])
    gpu = ak.to_backend(ak.Array(data), "cuda")
    result = complex(ak.var(gpu))
    assert result.imag == pytest.approx(0.0)
    assert result.real == pytest.approx(float(np.var(data)))
