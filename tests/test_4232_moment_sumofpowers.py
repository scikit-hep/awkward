# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

# ak.moment(x, n) accumulates the raw moment sum(x**n) in float64 directly from
# the input: n==2 via the sum-of-squares reducer, other n via the sum-of-powers
# reducer -- no x**n buffer and no integer/float32 overflow. Unweighted numeric
# cases need the awkward_reduce_sum{ofsquares,ofpowers} kernels (awkward-cpp).


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_moment_int32_overflow(n):
    # x**n overflows int32 for these values; float64 accumulation matches NumPy.
    data = np.array([1000, 2000, 3000], dtype=np.int32)
    result = ak.moment(ak.Array(data), n)
    assert result == pytest.approx(np.mean(data.astype(np.float64) ** n))


@pytest.mark.parametrize("n", [2, 3])
def test_moment_float32_precision(n):
    data = np.array([1e6, 2e6, 3e6], dtype=np.float32)
    result = ak.moment(ak.Array(data), n)
    assert result == pytest.approx(np.mean(data.astype(np.float64) ** n), rel=1e-6)


def test_moment_jagged():
    array = ak.values_astype(ak.Array([[1000, 2000, 3000], [], [4000]]), np.int32)
    out = ak.moment(array, 3, axis=-1)
    assert out[0] == pytest.approx(np.mean(np.array([1000.0, 2000.0, 3000.0]) ** 3))
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(4000.0**3)


def test_moment_weighted():
    x = np.array([1000, 2000, 3000], dtype=np.int32)
    w = np.array([1, 2, 3], dtype=np.int32)
    result = ak.moment(ak.Array(x), 3, weight=ak.Array(w))
    xf, wf = x.astype(np.float64), w.astype(np.float64)
    assert result == pytest.approx(np.sum(wf * xf**3) / np.sum(wf))


def test_moment_typetracer_is_float64():
    base = ak.values_astype(ak.Array([[1, 2, 3], [4, 5]]), np.int32)
    tt = ak.to_backend(base, "typetracer")
    assert str(ak.moment(tt, 3, axis=-1).type).endswith("float64")
