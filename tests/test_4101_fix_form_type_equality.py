# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward.forms.numpyform import NumpyForm
from awkward.forms.regularform import RegularForm
from awkward.types.listtype import ListType
from awkward.types.numpytype import NumpyType
from awkward.types.regulartype import RegularType


def test_numpyform_inner_shape_inequality():
    """NumpyForm with different inner_shapes must NOT compare equal."""
    a = NumpyForm("float64", (2,))
    b = NumpyForm("float64", (3,))
    assert a != b


def test_numpyform_inner_shape_unknown_length():
    """NumpyForm inner_shape with unknown_length is compatible with any size."""
    a = NumpyForm("float64", (unknown_length,))
    b = NumpyForm("float64", (3,))
    assert a == b


def test_numpyform_inner_shape_length_mismatch():
    """NumpyForm with different inner_shape ranks must NOT compare equal."""
    a = NumpyForm("float64", (2, 3))
    b = NumpyForm("float64", (2,))
    assert a != b


def test_regularform_unknown_length_no_typeerror():
    """RegularForm._is_equal_to must not raise TypeError with unknown_length size."""
    a = RegularForm(NumpyForm("float64"), unknown_length)
    b = RegularForm(NumpyForm("float64"), 3)
    assert a._is_equal_to(b, all_parameters=False, form_key=False)
    assert b._is_equal_to(a, all_parameters=False, form_key=False)


def test_regulartype_unknown_length_no_typeerror():
    """RegularType._is_equal_to must not raise TypeError with unknown_length size."""
    a = RegularType(NumpyType("float64"), unknown_length)
    b = RegularType(NumpyType("float64"), 3)
    assert a._is_equal_to(b, all_parameters=False)
    assert b._is_equal_to(a, all_parameters=False)


def test_listtype_all_parameters_propagation():
    """ListType._is_equal_to must propagate all_parameters to content comparison."""
    inner_with_params = NumpyType("float64", parameters={"custom": "value"})
    inner_without_params = NumpyType("float64")

    list_with = ListType(inner_with_params)
    list_without = ListType(inner_without_params)

    assert not list_with._is_equal_to(list_without, all_parameters=True)
    assert list_with._is_equal_to(list_without, all_parameters=False)


def test_union_purelist_parameters_second_key():
    """purelist_parameters must try all keys, not stop after the first misses."""
    arr1 = ak.with_parameter(ak.Array([[1, 2], [3]]), "__list__", "MyList")
    arr2 = ak.with_parameter(ak.Array([[4, 5], [6]]), "__list__", "MyList")
    union = ak.concatenate(
        [arr1[np.array([True, False])], arr2[np.array([False, True])]]
    )
    result = union.layout.purelist_parameters("__record__", "__list__")
    assert result == "MyList"


def test_union_purelist_parameters_inconsistent_returns_none():
    """purelist_parameters returns None when contents disagree on a key."""
    arr1 = ak.with_parameter(ak.Array([[1, 2], [3]]), "__list__", "ListA")
    arr2 = ak.with_parameter(ak.Array([[4, 5], [6]]), "__list__", "ListB")
    union = ak.concatenate(
        [arr1[np.array([True, False])], arr2[np.array([False, True])]]
    )
    result = union.layout.purelist_parameters("__list__")
    assert result is None
