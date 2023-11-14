# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

array = ak.Array([[0, 2, 3.0], [4, 5, 6, 7, 8], [], [9, 8, None], [10, 1], []])


def test_sum():
    assert ak.sum(array, axis=None) == pytest.approx(63.0)
    assert ak.almost_equal(
        ak.sum(array, axis=None, keepdims=True), ak.to_regular([[63.0]])
    )
    assert ak.almost_equal(
        ak.sum(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[63.0]]).mask[[[True]]]),
    )
    assert ak.sum(array[2], axis=None, mask_identity=True) is None


def test_prod():
    assert ak.prod(array[1:], axis=None) == pytest.approx(4838400.0)
    assert ak.prod(array, axis=None) == 0
    assert ak.almost_equal(
        ak.prod(array, axis=None, keepdims=True), ak.to_regular([[0.0]])
    )
    assert ak.almost_equal(
        ak.prod(array[1:], axis=None, keepdims=True), ak.to_regular([[4838400.0]])
    )
    assert ak.almost_equal(
        ak.prod(array[1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[4838400.0]]).mask[[[True]]]),
    )
    assert ak.prod(array[2], axis=None, mask_identity=True) is None


def test_min():
    assert ak.min(array, axis=None) == pytest.approx(0.0)
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[0.0]]),
    )
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, initial=-100, mask_identity=False),
        ak.to_regular([[-100.0]]),
    )

    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0.0]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.min(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[np.inf]]).mask[[[False]]]),
    )
    assert ak.min(array[2], axis=None, mask_identity=True) is None


def test_max():
    assert ak.max(array, axis=None) == pytest.approx(10.0)
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[10.0]]),
    )
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, initial=100, mask_identity=False),
        ak.to_regular([[100.0]]),
    )
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[10.0]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.max(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[np.inf]]).mask[[[False]]]),
    )
    assert ak.max(array[2], axis=None, mask_identity=True) is None


def test_count():
    assert ak.count(array, axis=None) == 12
    assert ak.almost_equal(
        ak.count(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[12]]),
    )
    assert ak.almost_equal(
        ak.count(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[12]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.count(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0]]).mask[[[False]]]),
    )
    assert ak.count(array[2], axis=None, mask_identity=True) is None
    assert ak.count(array[2], axis=None, mask_identity=False) == 0


def test_count_nonzero():
    assert ak.count_nonzero(array, axis=None) == 11
    assert ak.almost_equal(
        ak.count_nonzero(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[11]]),
    )
    assert ak.almost_equal(
        ak.count_nonzero(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[11]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.count_nonzero(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0]]).mask[[[False]]]),
    )
    assert ak.count_nonzero(array[2], axis=None, mask_identity=True) is None
    assert ak.count_nonzero(array[2], axis=None, mask_identity=False) == 0


def test_std():
    assert ak.std(array, axis=None) == pytest.approx(3.139134700306227)
    assert ak.almost_equal(
        ak.std(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[3.139134700306227]]),
    )
    assert ak.almost_equal(
        ak.std(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[3.139134700306227]]).mask[[[True]]]),
    )
    assert np.isnan(ak.std(array[2], axis=None, mask_identity=False))


def test_std_no_mask_axis_none():
    assert ak.almost_equal(
        ak.std(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0.0]]).mask[[[False]]]),
    )
    assert ak.std(array[2], axis=None, mask_identity=True) is None


def test_var():
    assert ak.var(array, axis=None) == pytest.approx(9.854166666666666)
    assert ak.almost_equal(
        ak.var(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[9.854166666666666]]),
    )
    assert ak.almost_equal(
        ak.var(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[9.854166666666666]]).mask[[[True]]]),
    )
    assert np.isnan(ak.var(array[2], axis=None, mask_identity=False))


def test_var_no_mask_axis_none():
    assert ak.almost_equal(
        ak.var(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0.0]]).mask[[[False]]]),
    )
    assert ak.var(array[2], axis=None, mask_identity=True) is None


def test_mean():
    assert ak.mean(array, axis=None) == pytest.approx(5.25)
    assert ak.almost_equal(
        ak.mean(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[5.25]]),
    )
    assert ak.almost_equal(
        ak.mean(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[5.25]]).mask[[[True]]]),
    )
    assert np.isnan(ak.mean(array[2], axis=None, mask_identity=False))


def test_mean_no_mask_axis_none():
    assert ak.almost_equal(
        ak.mean(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0.0]]).mask[[[False]]]),
    )
    assert ak.mean(array[2], axis=None, mask_identity=True) is None


def test_ptp():
    assert ak.ptp(array, axis=None) == pytest.approx(10.0)
    assert ak.almost_equal(
        ak.ptp(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[10.0]]),
    )
    assert ak.almost_equal(
        ak.ptp(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[10.0]]).mask[[[True]]]),
    )
    assert ak.ptp(array[2], axis=None, mask_identity=False) == pytest.approx(0.0)


def test_ptp_no_mask_axis_none():
    assert ak.almost_equal(
        ak.ptp(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0.0]]).mask[[[False]]]),
    )
    assert ak.ptp(array[2], axis=None, mask_identity=True) is None


def test_argmax():
    assert ak.argmax(array, axis=None) == 11
    assert ak.almost_equal(
        ak.argmax(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[11]]),
    )
    assert ak.almost_equal(
        ak.argmax(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[11]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.argmax(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0]]).mask[[[False]]]),
    )
    assert ak.argmax(array[2], axis=None, mask_identity=True) is None
    assert ak.argmax(array[2], axis=None, mask_identity=False) == -1


def test_argmin():
    assert ak.argmin(array, axis=None) == 0
    assert ak.almost_equal(
        ak.argmin(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[0]]),
    )
    assert ak.almost_equal(
        ak.argmin(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[0]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.argmin(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[999]]).mask[[[False]]]),
    )
    assert ak.argmin(array[2], axis=None, mask_identity=True) is None
    assert ak.argmin(array[2], axis=None, mask_identity=False) == -1


def test_any():
    assert ak.any(array, axis=None)
    assert ak.almost_equal(
        ak.any(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[True]]),
    )
    assert ak.almost_equal(
        ak.any(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[True]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.any(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[True]]).mask[[[False]]]),
    )
    assert ak.any(array[2], axis=None, mask_identity=True) is None
    assert not ak.any(array[2], axis=None, mask_identity=False)


def test_all():
    assert not ak.all(array, axis=None)
    assert ak.almost_equal(
        ak.all(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular([[False]]),
    )
    assert ak.almost_equal(
        ak.all(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[False]]).mask[[[True]]]),
    )
    assert ak.almost_equal(
        ak.all(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(ak.Array([[False]]).mask[[[False]]]),
    )
    assert ak.all(array[2], axis=None, mask_identity=True) is None
    assert ak.all(array[2], axis=None, mask_identity=False)


def test_prod_numpy():
    array_reg = ak.from_numpy(np.arange(7 * 5 * 3, dtype=np.int64).reshape((7, 5, 3)))
    result_reg = ak.from_numpy(np.array([[[5460]]], dtype=np.int64))

    assert ak.sum(array_reg, axis=None) == 5460
    assert ak.almost_equal(ak.sum(array_reg, axis=None, keepdims=True), result_reg)
    assert ak.almost_equal(
        ak.sum(array_reg, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(result_reg.mask[[[[True]]]], axis=None),
    )
