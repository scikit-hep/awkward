# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_var_integer_no_overflow():
    # int32 squares overflow int32 (100000**2 = 1e10 > 2**31 - 1)
    data = np.array([100000, 200000, 300000], dtype=np.int32)
    result = ak.var(ak.Array(data))
    expected = np.var(data.astype(np.float64))
    assert result == pytest.approx(expected)
    assert np.asarray(result).dtype == np.dtype(np.float64)


def test_std_integer_no_overflow():
    data = np.array([100000, 200000, 300000], dtype=np.int32)
    result = ak.std(ak.Array(data))
    expected = np.std(data.astype(np.float64))
    assert result == pytest.approx(expected)


def test_moment_integer_no_overflow():
    data = np.array([100000, 200000, 300000], dtype=np.int32)
    result = ak.moment(ak.Array(data), 2)
    expected = np.mean(data.astype(np.float64) ** 2)
    assert result == pytest.approx(expected)


def test_mean_weighted_integer_no_overflow():
    # x * weight overflows int32
    x = np.array([100000, 200000, 300000], dtype=np.int32)
    weight = np.array([100000, 200000, 300000], dtype=np.int32)
    result = ak.mean(ak.Array(x), weight=ak.Array(weight))
    xf, wf = x.astype(np.float64), weight.astype(np.float64)
    expected = np.sum(xf * wf) / np.sum(wf)
    assert result == pytest.approx(expected)


def test_var_jagged_integer_no_overflow():
    # int32 leaves, so x*x overflows int32 without the promotion (values entered
    # as the default int64 would only square to ~2.5e11 and pass on main too).
    array = ak.values_astype(
        ak.Array([[100000, 200000, 300000], [], [400000, 500000]]), np.int32
    )
    result = ak.var(array, axis=-1)
    expected = [
        np.var([100000.0, 200000.0, 300000.0]),
        np.nan,
        np.var([400000.0, 500000.0]),
    ]
    assert result[0] == pytest.approx(expected[0])
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(expected[2])


def test_float_input_unchanged():
    # Floating input must still work and match NumPy (helper is a no-op here).
    data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    assert ak.var(ak.Array(data)) == pytest.approx(np.var(data))
    assert ak.mean(ak.Array(data)) == pytest.approx(np.mean(data))


def test_var_int64_no_overflow_issue_3525():
    # Resolves #3525: the squares overflow int64 (the reducer's accumulator), so
    # on main the variance goes negative and ak.std returns nan. The new int32
    # tests do not exercise the int64 accumulator path.
    data = np.array([3_000_000_000, 4_000_000_000, 5_000_000_000], dtype=np.int64)
    result = ak.var(ak.Array(data))
    assert result == pytest.approx(np.var(data.astype(np.float64)))
    assert not np.isnan(ak.std(ak.Array(data)))


def test_typetracer_promotion():
    # The promotion must work on typetracer layouts (protects the dask-awkward
    # path): integer input still yields float64 output types.
    base = ak.values_astype(ak.Array([[1, 2, 3], [4, 5]]), np.int32)
    tt = ak.to_backend(base, "typetracer")
    assert str(ak.var(tt, axis=-1).type) == "2 * float64"
    assert str(ak.mean(tt, axis=-1).type) == "2 * float64"
    assert ak.moment(tt, 2, axis=-1).layout.dtype == np.dtype(np.float64)
