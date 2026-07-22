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
    assert np.dtype(np.asarray(result).dtype) == np.dtype(np.float64)


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
    array = ak.Array([[100000, 200000, 300000], [], [400000, 500000]])
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
