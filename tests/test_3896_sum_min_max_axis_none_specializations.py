# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.typetracer import MaybeNone, is_unknown_scalar


def test_sum_axis_none_non_empty():
    array = ak.Array([1, 2, 3, 4, 5])
    assert ak.sum(array, axis=None) == 15

    array = ak.Array([1.0, 2.0, 3.0])
    assert ak.sum(array, axis=None) == 6.0


def test_sum_axis_none_empty():
    array = ak.Array(np.array([], dtype=np.int64))
    assert ak.sum(array, axis=None) == 0

    array = ak.Array(np.array([], dtype=np.float64))
    assert ak.sum(array, axis=None) == 0.0


def test_min_axis_none_non_empty():
    int_array = ak.Array([3, 1, 4, 1, 5, 9, 2, 6])
    float_array = ak.Array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

    # basic (default mask_identity=True, non-empty so not masked)
    assert ak.min(int_array, axis=None) == 1
    assert ak.min(float_array, axis=None) == 1.0

    # mask_identity=False
    assert ak.min(int_array, axis=None, mask_identity=False) == 1
    assert ak.min(float_array, axis=None, mask_identity=False) == 1.0

    # mask_identity=True (non-empty, so result is not masked)
    assert ak.min(int_array, axis=None, mask_identity=True) == 1
    assert ak.min(float_array, axis=None, mask_identity=True) == 1.0

    # initial lower than array minimum → initial wins
    assert ak.min(int_array, axis=None, initial=0, mask_identity=False) == 0
    assert ak.min(float_array, axis=None, initial=0.0, mask_identity=False) == 0.0

    # initial higher than array minimum → array wins
    assert ak.min(int_array, axis=None, initial=2, mask_identity=False) == 1
    assert ak.min(float_array, axis=None, initial=2.0, mask_identity=False) == 1.0

    # initial with mask_identity=True (non-empty, so result is not masked)
    assert ak.min(int_array, axis=None, initial=0, mask_identity=True) == 0
    assert ak.min(float_array, axis=None, initial=0.0, mask_identity=True) == 0.0


def test_min_axis_none_empty():
    int_array = ak.Array(np.array([], dtype=np.int64))
    float_array = ak.Array(np.array([], dtype=np.float64))

    # default (mask_identity=True) → None
    assert ak.min(int_array, axis=None) is None
    assert ak.min(float_array, axis=None) is None

    # mask_identity=False → identity element
    assert ak.min(int_array, axis=None, mask_identity=False) == np.iinfo(np.int64).max
    assert ak.min(float_array, axis=None, mask_identity=False) == np.inf

    # initial with mask_identity=False → initial
    assert ak.min(int_array, axis=None, initial=5, mask_identity=False) == 5
    assert ak.min(float_array, axis=None, initial=5.0, mask_identity=False) == 5.0

    # initial with mask_identity=True (default) → still None
    assert ak.min(int_array, axis=None, initial=5) is None
    assert ak.min(float_array, axis=None, initial=5.0) is None


def test_max_axis_none_non_empty():
    int_array = ak.Array([3, 1, 4, 1, 5, 9, 2, 6])
    float_array = ak.Array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

    # basic (default mask_identity=True, non-empty so not masked)
    assert ak.max(int_array, axis=None) == 9
    assert ak.max(float_array, axis=None) == 9.0

    # mask_identity=False
    assert ak.max(int_array, axis=None, mask_identity=False) == 9
    assert ak.max(float_array, axis=None, mask_identity=False) == 9.0

    # mask_identity=True (non-empty, so result is not masked)
    assert ak.max(int_array, axis=None, mask_identity=True) == 9
    assert ak.max(float_array, axis=None, mask_identity=True) == 9.0

    # initial higher than array maximum → initial wins
    assert ak.max(int_array, axis=None, initial=10, mask_identity=False) == 10
    assert ak.max(float_array, axis=None, initial=10.0, mask_identity=False) == 10.0

    # initial lower than array maximum → array wins
    assert ak.max(int_array, axis=None, initial=2, mask_identity=False) == 9
    assert ak.max(float_array, axis=None, initial=2.0, mask_identity=False) == 9.0

    # initial with mask_identity=True (non-empty, so result is not masked)
    assert ak.max(int_array, axis=None, initial=10, mask_identity=True) == 10
    assert ak.max(float_array, axis=None, initial=10.0, mask_identity=True) == 10.0


def test_max_axis_none_empty():
    int_array = ak.Array(np.array([], dtype=np.int64))
    float_array = ak.Array(np.array([], dtype=np.float64))

    # default (mask_identity=True) → None
    assert ak.max(int_array, axis=None) is None
    assert ak.max(float_array, axis=None) is None

    # mask_identity=False → identity element
    assert ak.max(int_array, axis=None, mask_identity=False) == np.iinfo(np.int64).min
    assert ak.max(float_array, axis=None, mask_identity=False) == -np.inf

    # initial with mask_identity=False → initial
    assert ak.max(int_array, axis=None, initial=5, mask_identity=False) == 5
    assert ak.max(float_array, axis=None, initial=5.0, mask_identity=False) == 5.0

    # initial with mask_identity=True (default) → still None
    assert ak.max(int_array, axis=None, initial=5) is None
    assert ak.max(float_array, axis=None, initial=5.0) is None


@pytest.mark.parametrize("forget_length", [False, True])
def test_sum_axis_none_typetracer(forget_length):
    array = ak.Array([1, 2, 3, 4, 5])
    tt = ak.Array(array.layout.to_typetracer(forget_length=forget_length))

    result = ak.sum(tt, axis=None)
    assert not isinstance(result, MaybeNone)
    assert is_unknown_scalar(result)
    assert result.dtype == np.int64


@pytest.mark.parametrize("forget_length", [False, True])
def test_min_axis_none_typetracer(forget_length):
    array = ak.Array([3, 1, 4, 1, 5])
    tt = ak.Array(array.layout.to_typetracer(forget_length=forget_length))

    # mask_identity=True (default) → MaybeNone
    result = ak.min(tt, axis=None)
    assert isinstance(result, MaybeNone)
    assert is_unknown_scalar(result.content)
    assert result.content.dtype == np.int64

    # mask_identity=False → unknown scalar, not MaybeNone
    result = ak.min(tt, axis=None, mask_identity=False)
    assert not isinstance(result, MaybeNone)
    assert is_unknown_scalar(result)
    assert result.dtype == np.int64

    # initial with mask_identity=True → MaybeNone
    result = ak.min(tt, axis=None, initial=0)
    assert isinstance(result, MaybeNone)
    assert is_unknown_scalar(result.content)
    assert result.content.dtype == np.int64

    # initial with mask_identity=False → unknown scalar, not MaybeNone
    result = ak.min(tt, axis=None, initial=0, mask_identity=False)
    assert not isinstance(result, MaybeNone)
    assert is_unknown_scalar(result)
    assert result.dtype == np.int64


@pytest.mark.parametrize("forget_length", [False, True])
def test_max_axis_none_typetracer(forget_length):
    array = ak.Array([3, 1, 4, 1, 5])
    tt = ak.Array(array.layout.to_typetracer(forget_length=forget_length))

    # mask_identity=True (default) → MaybeNone
    result = ak.max(tt, axis=None)
    assert isinstance(result, MaybeNone)
    assert is_unknown_scalar(result.content)
    assert result.content.dtype == np.int64

    # mask_identity=False → unknown scalar, not MaybeNone
    result = ak.max(tt, axis=None, mask_identity=False)
    assert not isinstance(result, MaybeNone)
    assert is_unknown_scalar(result)
    assert result.dtype == np.int64

    # initial with mask_identity=True → MaybeNone
    result = ak.max(tt, axis=None, initial=10)
    assert isinstance(result, MaybeNone)
    assert is_unknown_scalar(result.content)
    assert result.content.dtype == np.int64

    # initial with mask_identity=False → unknown scalar, not MaybeNone
    result = ak.max(tt, axis=None, initial=10, mask_identity=False)
    assert not isinstance(result, MaybeNone)
    assert is_unknown_scalar(result)
    assert result.dtype == np.int64
