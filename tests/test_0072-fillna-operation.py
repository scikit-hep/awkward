# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_fillna_empty_array():
    empty = ak.contents.EmptyArray()
    value = ak.contents.NumpyArray(np.array([10]))

    assert to_list(empty) == []
    array = ak._do.pad_none(empty, 5, 0)
    assert to_list(array) == [None, None, None, None, None]
    assert to_list(ak._do.fill_none(array, value)) == [10, 10, 10, 10, 10]


def test_fillna_numpy_array():
    content = ak.contents.NumpyArray(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    value = ak.contents.NumpyArray(np.array([0]))

    array = ak._do.pad_none(content, 3, 0)
    assert to_list(array) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], None]
    assert to_list(ak._do.fill_none(array, value)) == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        0,
    ]

    array = ak._do.pad_none(content, 5, 1)
    assert to_list(array) == [
        [1.1, 2.2, 3.3, None, None],
        [4.4, 5.5, 6.6, None, None],
    ]

    assert to_list(ak._do.fill_none(array, value)) == [
        [1.1, 2.2, 3.3, 0, 0],
        [4.4, 5.5, 6.6, 0, 0],
    ]


def test_fillna_regular_array():
    content = ak.contents.NumpyArray(
        np.array(
            [
                2.1,
                8.4,
                7.4,
                1.6,
                2.2,
                3.4,
                6.2,
                5.4,
                1.5,
                3.9,
                3.8,
                3.0,
                8.5,
                6.9,
                4.3,
                3.6,
                6.7,
                1.8,
                3.2,
            ]
        )
    )
    index = ak.index.Index64(
        np.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=np.int64)
    )
    indexedarray = ak.contents.IndexedOptionArray(index, content)
    regarray = ak.contents.RegularArray(indexedarray, 3, zeros_length=0)
    value = ak.contents.NumpyArray(np.array([666]))

    assert to_list(regarray) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6], [3.6, None, 6.7]]

    assert to_list(ak._do.fill_none(regarray, value)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, 666, 6.7],
    ]


def test_fillna_listarray_array():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.index.Index64(np.array([0, 3, 4, 5, 8]))
    stops = ak.index.Index64(np.array([3, 3, 6, 8, 9]))
    listarray = ak.contents.ListArray(starts, stops, content)
    value = ak.contents.NumpyArray(np.array([55]))

    assert to_list(listarray) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]

    assert to_list(ak._do.fill_none(listarray, value)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]


def test_fillna_unionarray():
    content1 = ak.operations.from_iter([[], [1.1], [2.2, 2.2]], highlevel=False)
    content2 = ak.operations.from_iter([["two", "two"], ["one"], []], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.contents.UnionArray(tags, index, [content1, content2])

    assert to_list(array) == [[], ["two", "two"], [1.1], ["one"], [2.2, 2.2], []]

    padded_array = ak._do.pad_none(array, 2, 1)
    assert to_list(padded_array) == [
        [None, None],
        ["two", "two"],
        [1.1, None],
        ["one", None],
        [2.2, 2.2],
        [None, None],
    ]

    value = ak.contents.NumpyArray(np.array([777]))
    assert to_list(ak._do.fill_none(padded_array, value)) == [
        [777, 777],
        ["two", "two"],
        [1.1, 777],
        ["one", 777],
        [2.2, 2.2],
        [777, 777],
    ]


def test_highlevel():
    array = ak.highlevel.Array([[1.1, 2.2, None, 3.3], [], [4.4, None, 5.5]]).layout
    assert to_list(ak.operations.fill_none(array, 999, axis=1)) == [
        [1.1, 2.2, 999, 3.3],
        [],
        [4.4, 999, 5.5],
    ]
    assert to_list(ak.operations.fill_none(array, [1, 2, 3], axis=1)) == [
        [1.1, 2.2, [1, 2, 3], 3.3],
        [],
        [4.4, [1, 2, 3], 5.5],
    ]
    assert to_list(ak.operations.fill_none(array, [], axis=1)) == [
        [1.1, 2.2, [], 3.3],
        [],
        [4.4, [], 5.5],
    ]
    assert to_list(ak.operations.fill_none(array, {"x": 999}, axis=1)) == [
        [1.1, 2.2, {"x": 999}, 3.3],
        [],
        [4.4, {"x": 999}, 5.5],
    ]

    array = ak.highlevel.Array([[1.1, 2.2, 3.3], None, [], None, [4.4, 5.5]]).layout
    assert to_list(ak.operations.fill_none(array, 999, axis=0)) == [
        [1.1, 2.2, 3.3],
        999,
        [],
        999,
        [4.4, 5.5],
    ]
    assert to_list(ak.operations.fill_none(array, [1, 2, 3], axis=0)) == [
        [1.1, 2.2, 3.3],
        [1, 2, 3],
        [],
        [1, 2, 3],
        [4.4, 5.5],
    ]
    assert to_list(ak.operations.fill_none(array, {"x": 999}, axis=0)) == [
        [1.1, 2.2, 3.3],
        {"x": 999},
        [],
        {"x": 999},
        [4.4, 5.5],
    ]
    assert to_list(ak.operations.fill_none(array, [], axis=0)) == [
        [1.1, 2.2, 3.3],
        [],
        [],
        [],
        [4.4, 5.5],
    ]
