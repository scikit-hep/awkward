# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def tt(highlevel):
    return ak.highlevel.Array(highlevel.layout.to_typetracer())


def test_basic():
    array = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert to_list(array + array) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]
    assert (array + array).layout.form == (tt(array) + tt(array)).layout.form
    assert to_list(array * 2) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]
    assert to_list(2 * array) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]
    assert (array * 2).layout.form == (tt(array) * 2).layout.form
    assert (array * 2).layout.form == (2 * tt(array)).layout.form


def test_emptyarray():
    one = ak.highlevel.Array(ak.contents.NumpyArray(np.array([])))
    two = ak.highlevel.Array(ak.contents.EmptyArray())
    assert to_list(one + one) == []
    assert to_list(two + two) == []
    assert to_list(one + two) == []
    assert (one + one).layout.form == (tt(one) + tt(one)).layout.form
    assert (two + two).layout.form == (tt(two) + tt(two)).layout.form
    assert (one + two).layout.form == (tt(one) + tt(two)).layout.form


def test_indexedarray():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index1 = ak.index.Index64(np.array([2, 4, 4, 0, 8], dtype=np.int64))
    index2 = ak.index.Index64(np.array([6, 4, 4, 8, 0], dtype=np.int64))
    one = ak.highlevel.Array(ak.contents.IndexedArray(index1, content))
    two = ak.highlevel.Array(ak.contents.IndexedArray(index2, content))
    assert to_list(one + two) == [8.8, 8.8, 8.8, 8.8, 8.8]
    assert (one + two).layout.form == (tt(one) + tt(two)).layout.form


def test_indexedoptionarray():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index1 = ak.index.Index64(np.array([2, -1, 4, 0, 8], dtype=np.int64))
    index2 = ak.index.Index64(np.array([-1, 4, 4, -1, 0], dtype=np.int64))
    one = ak.highlevel.Array(ak.contents.IndexedOptionArray(index1, content))
    two = ak.highlevel.Array(ak.contents.IndexedOptionArray(index2, content))
    assert to_list(one + two) == [None, None, 8.8, None, 8.8]
    assert (one + two).layout.form == (tt(one) + tt(two)).layout.form

    uno = ak.highlevel.Array(
        ak.contents.NumpyArray(np.array([2.2, 4.4, 4.4, 0.0, 8.8]))
    )
    dos = ak.highlevel.Array(
        ak.contents.NumpyArray(np.array([6.6, 4.4, 4.4, 8.8, 0.0]))
    )
    assert to_list(uno + two) == [None, 8.8, 8.8, None, 8.8]
    assert (uno + two).layout.form == (tt(uno) + tt(two)).layout.form
    assert to_list(one + dos) == [8.8, None, 8.8, 8.8, 8.8]
    assert (one + dos).layout.form == (tt(one) + tt(dos)).layout.form


def test_regularize_shape():
    array = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert isinstance(array.to_RegularArray(), ak.contents.RegularArray)
    assert to_list(array.to_RegularArray()) == to_list(array)


def test_regulararray():
    array = ak.highlevel.Array(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert (
        to_list(array + array) == (np.arange(2 * 3 * 5).reshape(2, 3, 5) * 2).tolist()
    )
    assert (array + array).layout.form == (tt(array) + tt(array)).layout.form
    assert to_list(array * 2) == (np.arange(2 * 3 * 5).reshape(2, 3, 5) * 2).tolist()
    assert (array * 2).layout.form == (tt(array) * 2).layout.form
    array2 = ak.highlevel.Array(np.arange(2 * 1 * 5).reshape(2, 1, 5))
    assert to_list(array + array2) == to_list(
        np.arange(2 * 3 * 5).reshape(2, 3, 5) + np.arange(2 * 1 * 5).reshape(2, 1, 5)
    )
    assert (array + array2).layout.form == (tt(array) + tt(array2)).layout.form
    array3 = ak.highlevel.Array(np.arange(2 * 3 * 5).reshape(2, 3, 5).tolist())
    assert to_list(array + array3) == to_list(
        np.arange(2 * 3 * 5).reshape(2, 3, 5) + np.arange(2 * 3 * 5).reshape(2, 3, 5)
    )
    assert (array + array3).layout.form == (tt(array) + tt(array3)).layout.form
    assert to_list(array3 + array) == to_list(
        np.arange(2 * 3 * 5).reshape(2, 3, 5) + np.arange(2 * 3 * 5).reshape(2, 3, 5)
    )
    assert (array3 + array).layout.form == (tt(array3) + tt(array)).layout.form


def test_listarray():
    content = ak.contents.NumpyArray(np.arange(12, dtype=np.int64))
    starts = ak.index.Index64(np.array([3, 0, 999, 2, 6, 10], dtype=np.int64))
    stops = ak.index.Index64(np.array([7, 3, 999, 4, 6, 12], dtype=np.int64))
    one = ak.highlevel.Array(ak.contents.ListArray(starts, stops, content))
    two = ak.highlevel.Array(
        [[100, 100, 100, 100], [200, 200, 200], [], [300, 300], [], [400, 400]]
    )
    assert to_list(one) == [[3, 4, 5, 6], [0, 1, 2], [], [2, 3], [], [10, 11]]
    assert to_list(one + 100) == [
        [103, 104, 105, 106],
        [100, 101, 102],
        [],
        [102, 103],
        [],
        [110, 111],
    ]
    assert (one + 100).layout.form == (tt(one) + 100).layout.form
    assert to_list(one + two) == [
        [103, 104, 105, 106],
        [200, 201, 202],
        [],
        [302, 303],
        [],
        [410, 411],
    ]
    assert (one + two).layout.form == (tt(one) + tt(two)).layout.form
    assert to_list(two + one) == [
        [103, 104, 105, 106],
        [200, 201, 202],
        [],
        [302, 303],
        [],
        [410, 411],
    ]
    assert (two + one).layout.form == (tt(two) + tt(one)).layout.form
    assert to_list(one + np.array([100, 200, 300, 400, 500, 600])[:, np.newaxis]) == [
        [103, 104, 105, 106],
        [200, 201, 202],
        [],
        [402, 403],
        [],
        [610, 611],
    ]
    assert to_list(np.array([100, 200, 300, 400, 500, 600])[:, np.newaxis] + one) == [
        [103, 104, 105, 106],
        [200, 201, 202],
        [],
        [402, 403],
        [],
        [610, 611],
    ]
    assert to_list(one + 100) == [
        [103, 104, 105, 106],
        [100, 101, 102],
        [],
        [102, 103],
        [],
        [110, 111],
    ]
    assert (one + 100).layout.form == (tt(one) + 100).layout.form


def test_unionarray():
    one = ak.Array([1, 2, 3, [], [4, 5]])
    assert np.square(one).tolist() == [1, 4, 9, [], [16, 25]]

    two = ak.Array([[1.1], [2.2, 2.2], 3.3, 4.4, 5.5])

    uno, dos = ak.broadcast_arrays(one, two)
    assert uno.to_list() == [[1], [2, 2], 3, [], [4, 5]]
    assert dos.to_list() == [[1.1], [2.2, 2.2], 3.3, [], [5.5, 5.5]]
