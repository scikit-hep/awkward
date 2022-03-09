# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


def test_record():
    array = ak._v2.Array(
        [
            {"x": 10},
            {"x": 11},
            {"x": 12},
        ]
    )

    assert not ak._v2.is_tuple(array)
    assert not array.layout.is_tuple


def test_record_list():
    array = ak._v2.Array(
        [
            [
                {"x": 10},
                {"x": 11},
                {"x": 12},
            ]
        ]
    )

    assert not ak._v2.is_tuple(array)
    assert not array.layout.is_tuple


def test_tuple():
    array = ak._v2.Array(
        [
            (10,),
            (11,),
            (12,),
        ]
    )

    assert ak._v2.is_tuple(array)
    assert array.layout.is_tuple


def test_tuple_list():
    array = ak._v2.Array(
        [
            [
                (10,),
                (11,),
                (12,),
            ]
        ]
    )

    assert ak._v2.is_tuple(array)
    assert array.layout.is_tuple


def test_list():
    array = ak._v2.Array([[10, 11, 12]])

    assert not ak._v2.is_tuple(array)
    assert not array.layout.is_tuple


def test_record_tuple():
    array = ak._v2.Array([{"x": (10,)}])

    assert not ak._v2.is_tuple(array)
    assert not array.layout.is_tuple


def test_tuple_record():
    array = ak._v2.Array([({"x": 10},)])

    assert ak._v2.is_tuple(array)
    assert array.layout.is_tuple
