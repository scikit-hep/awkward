# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_empty_array_slice():
    # inspired by PR021::test_getitem
    a = ak._v2.operations.from_json("[[], [[], []], [[], [], []]]")
    assert to_list(a[2, 1, np.array([], dtype=np.int64)]) == []
    # FIXME: assert [[]] == []
    # assert to_list(a[2, np.array([1], dtype=np.int64), np.array([], dtype=np.int64)]) == []

    a = ak._v2.operations.from_iter([[], [[], []], [[], [], []]], highlevel=False)
    assert to_list(a[2, 1, np.array([], dtype=np.int64)]) == []
    assert (
        a.typetracer[2, 1, np.array([], dtype=np.int64)].form
        == a[2, 1, np.array([], dtype=np.int64)].form
    )
    assert to_list(a[2, np.array([1], dtype=int), np.array([], dtype=int)]) == []
    assert (
        a.typetracer[
            2, np.array([1], dtype=np.int64), np.array([], dtype=np.int64)
        ].form
        == a[2, np.array([1], dtype=np.int64), np.array([], dtype=np.int64)].form
    )

    # inspired by PR015::test_deep_numpy
    content = ak._v2.contents.NumpyArray(
        np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
    )
    listarray = ak._v2.contents.ListArray(
        ak._v2.index.Index64(np.array([0, 3, 3])),
        ak._v2.index.Index64(np.array([3, 3, 5])),
        content,
    )
    assert to_list(listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]]) == [
        8.8,
        5.5,
        0.0,
        7.7,
    ]
    assert (
        listarray.typetracer[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]].form
        == listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]].form
    )
    assert to_list(listarray[2, 1, np.array([], dtype=np.int64)]) == []
    assert (
        listarray.typetracer[2, 1, np.array([], dtype=np.int64)].form
        == listarray[2, 1, np.array([], dtype=np.int64)].form
    )
    assert to_list(listarray[2, 1, []]) == []
    assert listarray.typetracer[2, 1, []].form == listarray[2, 1, []].form
    assert to_list(listarray[2, [1], []]) == []
    assert listarray.typetracer[2, [1], []].form == listarray[2, [1], []].form
    assert to_list(listarray[2, [], []]) == []
    assert listarray.typetracer[2, [], []].form == listarray[2, [], []].form


def test_nonflat_slice():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)

    content = ak._v2.contents.NumpyArray(array.reshape(-1))
    inneroffsets = ak._v2.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak._v2.index.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak._v2.contents.ListOffsetArray(
        outeroffsets, ak._v2.contents.ListOffsetArray(inneroffsets, content)
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
        listoffsetarray.typetracer[
            [1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]
        ].form
        == one.form
    )


def test_nonflat_slice_2():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    content = ak._v2.contents.NumpyArray(array.reshape(-1))
    inneroffsets = ak._v2.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak._v2.index.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak._v2.contents.ListOffsetArray(
        outeroffsets, ak._v2.contents.ListOffsetArray(inneroffsets, content)
    )

    two = listoffsetarray[
        np.asarray([[1, 0], [1, 1], [1, 0]]),
        np.asarray([[2, 0], [1, 1], [2, 0]]),
        np.asarray([[2, 4], [2, 4], [0, 1]]),
    ]
    assert to_list(two) == [[27, 4], [22, 24], [25, 1]]


def test_newaxis():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)

    content = ak._v2.contents.NumpyArray(array.reshape(-1))
    inneroffsets = ak._v2.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak._v2.index.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak._v2.contents.ListOffsetArray(
        outeroffsets, ak._v2.contents.ListOffsetArray(inneroffsets, content)
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
        listoffsetarray.typetracer[:, np.newaxis].form
        == listoffsetarray[:, np.newaxis].form
    )
