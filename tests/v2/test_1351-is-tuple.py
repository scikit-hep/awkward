# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np


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


def test_union_tuple_int():
    array = ak._v2.Array([(10,), 20])

    assert not ak._v2.is_tuple(array)
    assert not array.layout.is_tuple


def test_union_tuple_tuple():
    array = ak._v2.Array([(10,), (20, 30)])

    assert ak._v2.is_tuple(array)
    assert array.layout.is_tuple


def test_indexed_tuple():
    array = ak._v2.Array(
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
            ak._v2.contents.RecordArray(
                [ak._v2.contents.NumpyArray(np.arange(10))], None
            ),
        )
    )

    assert ak._v2.is_tuple(array)
    assert array.layout.is_tuple


def test_indexed_record():
    array = ak._v2.Array(
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
            ak._v2.contents.RecordArray(
                [ak._v2.contents.NumpyArray(np.arange(10))], ["x"]
            ),
        )
    )

    assert not ak._v2.is_tuple(array)
    assert not array.layout.is_tuple
