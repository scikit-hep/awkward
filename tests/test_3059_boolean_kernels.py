# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/master/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak

to_list = ak.operations.to_list


def test_bool_RecordArray():
    array = ak.highlevel.Array(
        [
            {"x": True, "y": [True]},
            {"x": False, "y": [True, False]},
        ]
    )

    assert ak._do.is_unique(array.layout) is False
    assert ak._do.is_unique(array["x"].layout) is True
    assert ak._do.is_unique(array["y"].layout) is False


def test_bool_UnionArray():
    content1 = ak.operations.from_iter([[], [False], [True, True]], highlevel=False)
    content2 = ak.operations.from_iter([[False, False], [True], []], highlevel=False)
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.contents.UnionArray.simplified(tags, index, [content1, content2])

    assert to_list(array) == [
        [],
        [False, False],
        [False],
        [True],
        [True, True],
        [],
    ]
    assert ak._do.is_unique(array) is False
    assert to_list(ak._do.unique(array, axis=None)) == [False, True]
    assert to_list(ak._do.unique(array, axis=-1)) == [
        [],
        [False],
        [False],
        [True],
        [True],
        [],
    ]


def test_bool_IndexedArray():
    content = ak.from_iter([[True], [False]], highlevel=False)
    index = ak.index.Index64(np.array([1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)

    assert ak._do.is_unique(indexedarray) is True

    listoffsetarray = ak.operations.from_iter([[True], [False]], highlevel=False)

    index = ak.index.Index64(np.array([1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, listoffsetarray)
    assert to_list(indexedarray) == [
        [False],
        [True],
    ]
    assert ak._do.is_unique(indexedarray) is True

    assert ak._do.is_unique(indexedarray) is True
    assert to_list(ak._do.unique(indexedarray)) == [
        False,
        True,
    ]
    assert to_list(ak._do.unique(indexedarray, axis=-1)) == [
        [False],
        [True],
    ]


def test_bool_subranges_equal():
    array = ak.contents.NumpyArray(
        np.array(
            [
                [True, False, True, True, False],
                [False, False, True, True, True],
                [False, True, False, True, True],
            ]
        )
    )

    starts = ak.index.Index64(np.array([0, 5, 10]))
    stops = ak.index.Index64(np.array([5, 10, 15]))

    result = ak.sort(array, axis=-1, highlevel=False).content._subranges_equal(
        starts, stops, 15
    )
    assert result is True

    starts = ak.index.Index64(np.array([0, 7]))
    stops = ak.index.Index64(np.array([7, 15]))

    assert (
        ak.sort(array, axis=-1, highlevel=False).content._subranges_equal(
            starts, stops, 15
        )
        is False
    )

    starts = ak.index.Index64(np.array([0]))
    stops = ak.index.Index64(np.array([15]))

    assert (
        ak.sort(array, axis=-1, highlevel=False).content._subranges_equal(
            starts, stops, 15
        )
        is False
    )
