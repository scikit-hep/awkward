# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


def test_record():
    array = ak.Array(
        [
            {"x": 10},
            {"x": 11},
            {"x": 12},
        ]
    )

    assert not ak.is_tuple(array)
    assert not array.layout.istuple


def test_record_list():
    array = ak.Array(
        [
            [
                {"x": 10},
                {"x": 11},
                {"x": 12},
            ]
        ]
    )

    assert not ak.is_tuple(array)
    assert not array.layout.istuple


def test_tuple():
    array = ak.Array(
        [
            (10,),
            (11,),
            (12,),
        ]
    )

    assert ak.is_tuple(array)
    assert array.layout.istuple


def test_tuple_list():
    array = ak.Array(
        [
            [
                (10,),
                (11,),
                (12,),
            ]
        ]
    )

    assert ak.is_tuple(array)
    assert array.layout.istuple


def test_list():
    array = ak.Array([[10, 11, 12]])

    assert not ak.is_tuple(array)
    assert not array.layout.istuple


def test_record_tuple():
    array = ak.Array([{"x": (10,)}])

    assert not ak.is_tuple(array)
    assert not array.layout.istuple


def test_tuple_record():
    array = ak.Array([({"x": 10},)])

    assert ak.is_tuple(array)
    assert array.layout.istuple


def test_union_tuple_int():
    array = ak.Array([(10,), 20])

    assert not ak.is_tuple(array)
    assert not array.layout.istuple


def test_union_tuple_tuple():
    array = ak.Array([(10,), (20, 30)])

    assert ak.is_tuple(array)
    assert array.layout.istuple
