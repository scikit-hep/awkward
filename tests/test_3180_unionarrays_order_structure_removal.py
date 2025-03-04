# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import pytest

import awkward as ak


def test_ravel_boolean():
    x = ak.Array([None, [False], [True]])
    y = ak.ravel(ak.fill_none(x, False, axis=0))
    assert y.tolist() == [False, False, True]


def test_ravel_boolean_typetracer():
    x = ak.Array([None, [False], [True]], backend="typetracer")
    y = ak.ravel(ak.fill_none(x, False, axis=0))
    assert y.typestr == "## * bool"


def test_ravel_ListOffsetArray():
    a = ak.Array([[1, 2, 3], 4, [5, 6], 7, 8])
    b = ak.ravel(a)
    assert b.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_ravel_ListOffsetArray_typetracer():
    a = ak.Array([[1, 2, 3], 4, [5, 6], 7, 8], backend="typetracer")
    b = ak.ravel(a)
    assert b.typestr == "## * int64"


def test_ravel_nested_ListOffsetArrays():
    a = ak.Array([[[1, 2], 3], 4, [5, [[[6]]]], 7, 8])
    b = ak.ravel(a)
    assert b.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_ravel_nested_ListOffsetArrays_typetracer():
    a = ak.Array([[[1, 2], 3], 4, [5, [[[6]]]], 7, 8], backend="typetracer")
    b = ak.ravel(a)
    assert b.typestr == "## * int64"


def test_ravel_incompatible_contents():
    a = ak.Array([1, "high", 2, "low"])
    with pytest.raises(
        AssertionError,
        match="cannot merge NumpyArray with ListOffsetArray",
    ):
        ak.ravel(a)


def test_ravel_incompatible_contents_typetracer():
    a = ak.Array([1, "high", 2, "low"], backend="typetracer")
    with pytest.raises(
        AssertionError,
        match="cannot merge NumpyArray with ListOffsetArray",
    ):
        ak.ravel(a)
