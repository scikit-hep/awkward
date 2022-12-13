# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_empty_array_slice():
    # inspired by PR021::test_getitem
    a = ak.operations.from_json("[[], [[], []], [[], [], []]]")
    assert to_list(a[2, 1, np.array([], dtype=np.int64)]) == []
    # FIXME: assert [[]] == []
    # assert to_list(a[2, np.array([1], dtype=np.int64), np.array([], dtype=np.int64)]) == []

    a = ak.operations.from_iter([[], [[], []], [[], [], []]], highlevel=False)
    assert to_list(a[2, 1, np.array([], dtype=np.int64)]) == []
    assert (
        a.to_typetracer()[2, 1, np.array([], dtype=np.int64)].form
        == a[2, 1, np.array([], dtype=np.int64)].form
    )
    assert to_list(a[2, np.array([1], dtype=int), np.array([], dtype=int)]) == []
    assert (
        a.to_typetracer()[
            2, np.array([1], dtype=np.int64), np.array([], dtype=np.int64)
        ].form
        == a[2, np.array([1], dtype=np.int64), np.array([], dtype=np.int64)].form
    )

    # inspired by PR015::test_deep_numpy
    content = ak.contents.NumpyArray(
        np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
    )
    listarray = ak.contents.ListArray(
        ak.index.Index64(np.array([0, 3, 3])),
        ak.index.Index64(np.array([3, 3, 5])),
        content,
    )
    assert to_list(listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]]) == [
        8.8,
        5.5,
        0.0,
        7.7,
    ]
    assert (
        listarray.to_typetracer()[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]].form
        == listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]].form
    )
    assert to_list(listarray[2, 1, np.array([], dtype=np.int64)]) == []
    assert (
        listarray.to_typetracer()[2, 1, np.array([], dtype=np.int64)].form
        == listarray[2, 1, np.array([], dtype=np.int64)].form
    )
    assert to_list(listarray[2, 1, []]) == []
    assert listarray.to_typetracer()[2, 1, []].form == listarray[2, 1, []].form
    assert to_list(listarray[2, [1], []]) == []
    assert listarray.to_typetracer()[2, [1], []].form == listarray[2, [1], []].form
    assert to_list(listarray[2, [], []]) == []
    assert listarray.to_typetracer()[2, [], []].form == listarray[2, [], []].form


def test_nonflat_slice():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)

    content = ak.contents.NumpyArray(array.reshape(-1))
    inneroffsets = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.index.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.contents.ListOffsetArray(
        outeroffsets, ak.contents.ListOffsetArray(inneroffsets, content)
    )

    assert to_list(
        array[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]
    ) == [27, 4, 22, 24, 25, 1]
    assert to_list(
        array[
            [[1, 0], [1, 1], [1, 0]], [[2, 0], [1, 1], [2, 0]], [[2, 4], [2, 4], [0, 1]]
        ]
    ) == [[27, 4], [22, 24], [25, 1]]

    one = listoffsetarray[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]
    assert to_list(one) == [27, 4, 22, 24, 25, 1]
    assert (
        listoffsetarray.to_typetracer()[
            [1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]
        ].form
        == one.form
    )


def test_nonflat_slice_2():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    content = ak.contents.NumpyArray(array.reshape(-1))
    inneroffsets = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.index.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.contents.ListOffsetArray(
        outeroffsets, ak.contents.ListOffsetArray(inneroffsets, content)
    )

    two = listoffsetarray[
        np.asarray([[1, 0], [1, 1], [1, 0]]),
        np.asarray([[2, 0], [1, 1], [2, 0]]),
        np.asarray([[2, 4], [2, 4], [0, 1]]),
    ]
    assert to_list(two) == [[27, 4], [22, 24], [25, 1]]


def test_newaxis():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)

    content = ak.contents.NumpyArray(array.reshape(-1))
    inneroffsets = ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.index.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.contents.ListOffsetArray(
        outeroffsets, ak.contents.ListOffsetArray(inneroffsets, content)
    )

    assert to_list(array[:, np.newaxis]) == [
        [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]],
        [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]],
    ]

    assert to_list(listoffsetarray[:, np.newaxis]) == [
        [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]],
        [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]],
    ]
    assert (
        listoffsetarray.to_typetracer()[:, np.newaxis].form
        == listoffsetarray[:, np.newaxis].form
    )
