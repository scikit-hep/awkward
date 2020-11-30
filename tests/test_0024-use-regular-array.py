# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_empty_array_slice():
    # inspired by PR021::test_getitem
    a = ak.from_json("[[], [[], []], [[], [], []]]")
    assert ak.to_list(a[2, 1, np.array([], dtype=int)]) == []
    assert ak.to_list(a[2, np.array([1], dtype=int), np.array([], dtype=int)]) == []

    # inspired by PR015::test_deep_numpy
    content = ak.layout.NumpyArray(
        np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]])
    )
    listarray = ak.layout.ListArray64(
        ak.layout.Index64(np.array([0, 3, 3])),
        ak.layout.Index64(np.array([3, 3, 5])),
        content,
    )
    assert ak.to_list(listarray[[2, 0, 0, -1], [1, -1, 0, 0], [0, 1, 0, 1]]) == [
        8.8,
        5.5,
        0.0,
        7.7,
    ]
    assert ak.to_list(listarray[2, 1, np.array([], dtype=int)]) == []
    assert ak.to_list(listarray[2, 1, []]) == []
    assert ak.to_list(listarray[2, [1], []]) == []
    assert ak.to_list(listarray[2, [], []]) == []


def test_nonflat_slice():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    numpyarray = ak.layout.NumpyArray(array)

    content = ak.layout.NumpyArray(array.reshape(-1))
    inneroffsets = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.layout.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.layout.ListOffsetArray64(
        outeroffsets, ak.layout.ListOffsetArray64(inneroffsets, content)
    )
    listoffsetarray.setidentities()

    assert ak.to_list(
        array[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]
    ) == [27, 4, 22, 24, 25, 1]
    assert ak.to_list(
        array[
            [[1, 0], [1, 1], [1, 0]], [[2, 0], [1, 1], [2, 0]], [[2, 4], [2, 4], [0, 1]]
        ]
    ) == [[27, 4], [22, 24], [25, 1]]

    one = listoffsetarray[[1, 0, 1, 1, 1, 0], [2, 0, 1, 1, 2, 0], [2, 4, 2, 4, 0, 1]]
    assert ak.to_list(one) == [27, 4, 22, 24, 25, 1]
    assert np.asarray(one.identities).tolist() == [
        [1, 2, 2],
        [0, 0, 4],
        [1, 1, 2],
        [1, 1, 4],
        [1, 2, 0],
        [0, 0, 1],
    ]

    two = listoffsetarray[
        [[1, 0], [1, 1], [1, 0]], [[2, 0], [1, 1], [2, 0]], [[2, 4], [2, 4], [0, 1]]
    ]
    assert ak.to_list(two) == [[27, 4], [22, 24], [25, 1]]
    assert np.asarray(two.content.identities).tolist() == [
        [1, 2, 2],
        [0, 0, 4],
        [1, 1, 2],
        [1, 1, 4],
        [1, 2, 0],
        [0, 0, 1],
    ]
    assert two.identities is None


def test_newaxis():
    array = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    numpyarray = ak.layout.NumpyArray(array)

    content = ak.layout.NumpyArray(array.reshape(-1))
    inneroffsets = ak.layout.Index64(np.array([0, 5, 10, 15, 20, 25, 30]))
    outeroffsets = ak.layout.Index64(np.array([0, 3, 6]))
    listoffsetarray = ak.layout.ListOffsetArray64(
        outeroffsets, ak.layout.ListOffsetArray64(inneroffsets, content)
    )

    assert ak.to_list(array[:, np.newaxis]) == [
        [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]],
        [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]],
    ]

    assert ak.to_list(listoffsetarray[:, np.newaxis]) == [
        [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]],
        [[[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]],
    ]
