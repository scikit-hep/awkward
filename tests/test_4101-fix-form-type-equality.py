# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward.forms.numpyform import NumpyForm
from awkward.forms.regularform import RegularForm
from awkward.types.listtype import ListType
from awkward.types.numpytype import NumpyType
from awkward.types.regulartype import RegularType

# ---------------------------------------------------------------------------
# Fix 1: NumpyForm inner_shape inequality
# ---------------------------------------------------------------------------


def test_numpyform_inner_shape_inequality():
    """NumpyForm with different inner_shapes must NOT compare equal."""
    a = NumpyForm("float64", (2,))
    b = NumpyForm("float64", (3,))
    assert not a._is_equal_to(b, all_parameters=False, form_key=False)
    assert a != b


def test_numpyform_inner_shape_equality():
    """NumpyForm with matching inner_shapes must compare equal."""
    a = NumpyForm("float64", (2,))
    b = NumpyForm("float64", (2,))
    assert a._is_equal_to(b, all_parameters=False, form_key=False)
    assert a == b


def test_numpyform_inner_shape_unknown_length():
    """NumpyForm inner_shape with unknown_length is compatible with any size."""
    a = NumpyForm("float64", (unknown_length,))
    b = NumpyForm("float64", (3,))
    assert a._is_equal_to(b, all_parameters=False, form_key=False)
    assert a == b


def test_numpyform_inner_shape_length_mismatch():
    """NumpyForm with different inner_shape lengths must NOT compare equal."""
    a = NumpyForm("float64", (2, 3))
    b = NumpyForm("float64", (2,))
    assert not a._is_equal_to(b, all_parameters=False, form_key=False)
    assert a != b


# ---------------------------------------------------------------------------
# Fix 2: RegularForm / RegularType unknown_length comparison
# ---------------------------------------------------------------------------


def test_regularform_unknown_length_no_typeerror():
    """RegularForm._is_equal_to must not raise TypeError with unknown_length size."""
    a = RegularForm(NumpyForm("float64"), unknown_length)
    b = RegularForm(NumpyForm("float64"), 3)
    # Should not raise, and should be considered equal (unknown is compatible)
    assert a._is_equal_to(b, all_parameters=False, form_key=False)
    assert b._is_equal_to(a, all_parameters=False, form_key=False)


def test_regularform_known_size_inequality():
    """RegularForm with different known sizes must NOT compare equal."""
    a = RegularForm(NumpyForm("float64"), 2)
    b = RegularForm(NumpyForm("float64"), 3)
    assert not a._is_equal_to(b, all_parameters=False, form_key=False)


def test_regulartype_unknown_length_no_typeerror():
    """RegularType._is_equal_to must not raise TypeError with unknown_length size."""
    a = RegularType(NumpyType("float64"), unknown_length)
    b = RegularType(NumpyType("float64"), 3)
    assert a._is_equal_to(b, all_parameters=False)
    assert b._is_equal_to(a, all_parameters=False)


def test_regulartype_known_size_inequality():
    """RegularType with different known sizes must NOT compare equal."""
    a = RegularType(NumpyType("float64"), 2)
    b = RegularType(NumpyType("float64"), 3)
    assert not a._is_equal_to(b, all_parameters=False)


# ---------------------------------------------------------------------------
# Fix 3: ListType all_parameters flag propagation
# ---------------------------------------------------------------------------


def test_listtype_all_parameters_propagation():
    """ListType._is_equal_to must propagate all_parameters to content comparison."""
    inner_with_params = NumpyType("float64", parameters={"custom": "value"})
    inner_without_params = NumpyType("float64")

    list_with = ListType(inner_with_params)
    list_without = ListType(inner_without_params)

    # With all_parameters=True the contents differ -> not equal
    assert not list_with._is_equal_to(list_without, all_parameters=True)
    # With all_parameters=False type-relevant parameters are compared; custom params differ
    # but "custom" is not a type-relevant parameter, so they should be equal
    assert list_with._is_equal_to(list_without, all_parameters=False)


# ---------------------------------------------------------------------------
# Fix 4: UnionMeta.purelist_parameters — __list__ key resolved after __record__
# ---------------------------------------------------------------------------


def test_union_purelist_parameters_second_key():
    """purelist_parameters must try all keys, not just the first."""
    import numpy as np

    # Build a union of two arrays, each with __list__ parameter
    arr1 = ak.with_parameter(ak.Array([[1, 2], [3]]), "__list__", "MyList")
    arr2 = ak.with_parameter(ak.Array([[4, 5], [6]]), "__list__", "MyList")
    union = ak.concatenate(
        [arr1[np.array([True, False])], arr2[np.array([False, True])]]
    )
    # union is now a UnionArray; both contents have __list__=MyList
    layout = union.layout
    # purelist_parameters("__record__", "__list__") should find __list__
    result = layout.purelist_parameters("__record__", "__list__")
    assert result == "MyList", f"Expected 'MyList', got {result!r}"


def test_union_purelist_parameters_first_key_wins():
    """purelist_parameters returns the first matching key."""
    import numpy as np

    arr1 = ak.with_parameter(
        ak.with_parameter(ak.Array([[1, 2], [3]]), "__list__", "MyList"),
        "__record__",
        "MyRecord",
    )
    arr2 = ak.with_parameter(
        ak.with_parameter(ak.Array([[4, 5], [6]]), "__list__", "MyList"),
        "__record__",
        "MyRecord",
    )
    union = ak.concatenate(
        [arr1[np.array([True, False])], arr2[np.array([False, True])]]
    )
    layout = union.layout
    result = layout.purelist_parameters("__record__", "__list__")
    assert result == "MyRecord"


def test_union_purelist_parameters_inconsistent_returns_none():
    """purelist_parameters returns None when contents disagree on a key."""
    import numpy as np

    arr1 = ak.with_parameter(ak.Array([[1, 2], [3]]), "__list__", "ListA")
    arr2 = ak.with_parameter(ak.Array([[4, 5], [6]]), "__list__", "ListB")
    union = ak.concatenate(
        [arr1[np.array([True, False])], arr2[np.array([False, True])]]
    )
    layout = union.layout
    result = layout.purelist_parameters("__list__")
    assert result is None
