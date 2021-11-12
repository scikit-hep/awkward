# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_basic():
    array = ak._v2.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert ak.to_list(array + array) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]
    assert ak.to_list(array * 2) == [[2.2, 4.4, 6.6], [], [8.8, 11.0]]


def test_emptyarray():
    one = ak._v2.highlevel.Array(ak._v2.contents.NumpyArray(np.array([])))
    two = ak._v2.highlevel.Array(ak._v2.contents.EmptyArray())
    assert ak.to_list(one + one) == []
    assert ak.to_list(two + two) == []
    assert ak.to_list(one + two) == []


def test_indexedarray():
    content = ak._v2.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index1 = ak._v2.index.Index64(np.array([2, 4, 4, 0, 8], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([6, 4, 4, 8, 0], dtype=np.int64))
    one = ak._v2.highlevel.Array(ak._v2.contents.IndexedArray(index1, content))
    two = ak._v2.highlevel.Array(ak._v2.contents.IndexedArray(index2, content))
    assert ak.to_list(one + two) == [8.8, 8.8, 8.8, 8.8, 8.8]


def test_indexedoptionarray():
    content = ak._v2.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index1 = ak._v2.index.Index64(np.array([2, -1, 4, 0, 8], dtype=np.int64))
    index2 = ak._v2.index.Index64(np.array([-1, 4, 4, -1, 0], dtype=np.int64))
    one = ak._v2.highlevel.Array(ak._v2.contents.IndexedOptionArray(index1, content))
    two = ak._v2.highlevel.Array(ak._v2.contents.IndexedOptionArray(index2, content))
    assert ak.to_list(one + two) == [None, None, 8.8, None, 8.8]

    uno = ak._v2.contents.NumpyArray(np.array([2.2, 4.4, 4.4, 0.0, 8.8]))
    dos = ak._v2.contents.NumpyArray(np.array([6.6, 4.4, 4.4, 8.8, 0.0]))
    assert ak.to_list(uno + two) == [None, 8.8, 8.8, None, 8.8]
    assert ak.to_list(one + dos) == [8.8, None, 8.8, 8.8, 8.8]


def test_regularize_shape():
    array = ak._v2.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert isinstance(array.toRegularArray(), ak._v2.contents.RegularArray)
    assert ak.to_list(array.toRegularArray()) == ak.to_list(array)


def test_regulararray():
    array = ak._v2.highlevel.Array(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert (
        ak.to_list(array + array)
        == (np.arange(2 * 3 * 5).reshape(2, 3, 5) * 2).tolist()
    )
    assert ak.to_list(array * 2) == (np.arange(2 * 3 * 5).reshape(2, 3, 5) * 2).tolist()
    array2 = ak._v2.highlevel.Array(np.arange(2 * 1 * 5).reshape(2, 1, 5))
    assert ak.to_list(array + array2) == ak.to_list(
        np.arange(2 * 3 * 5).reshape(2, 3, 5) + np.arange(2 * 1 * 5).reshape(2, 1, 5)
    )
    array3 = ak._v2.highlevel.Array(np.arange(2 * 3 * 5).reshape(2, 3, 5).tolist())
    assert ak.to_list(array + array3) == ak.to_list(
        np.arange(2 * 3 * 5).reshape(2, 3, 5) + np.arange(2 * 3 * 5).reshape(2, 3, 5)
    )
    assert ak.to_list(array3 + array) == ak.to_list(
        np.arange(2 * 3 * 5).reshape(2, 3, 5) + np.arange(2 * 3 * 5).reshape(2, 3, 5)
    )


def test_listarray():
    content = ak._v2.contents.NumpyArray(np.arange(12, dtype=np.int64))
    starts = ak._v2.index.Index64(np.array([3, 0, 999, 2, 6, 10], dtype=np.int64))
    stops = ak._v2.index.Index64(np.array([7, 3, 999, 4, 6, 12], dtype=np.int64))
    one = ak._v2.highlevel.Array(ak._v2.contents.ListArray(starts, stops, content))
    two = ak._v2.highlevel.Array(
        [[100, 100, 100, 100], [200, 200, 200], [], [300, 300], [], [400, 400]]
    )
    assert ak.to_list(one) == [[3, 4, 5, 6], [0, 1, 2], [], [2, 3], [], [10, 11]]
    assert ak.to_list(one + 100) == [
        [103, 104, 105, 106],
        [100, 101, 102],
        [],
        [102, 103],
        [],
        [110, 111],
    ]
    assert ak.to_list(one + two) == [
        [103, 104, 105, 106],
        [200, 201, 202],
        [],
        [302, 303],
        [],
        [410, 411],
    ]
    assert ak.to_list(two + one) == [
        [103, 104, 105, 106],
        [200, 201, 202],
        [],
        [302, 303],
        [],
        [410, 411],
    ]
    assert ak.to_list(
        one + np.array([100, 200, 300, 400, 500, 600])[:, np.newaxis]
    ) == [[103, 104, 105, 106], [200, 201, 202], [], [402, 403], [], [610, 611]]
    assert ak.to_list(
        np.array([100, 200, 300, 400, 500, 600])[:, np.newaxis] + one
    ) == [[103, 104, 105, 106], [200, 201, 202], [], [402, 403], [], [610, 611]]
    assert ak.to_list(one + 100) == [
        [103, 104, 105, 106],
        [100, 101, 102],
        [],
        [102, 103],
        [],
        [110, 111],
    ]


def test_unionarray():
    one0 = ak._v2.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64))
    one1 = ak._v2.contents.NumpyArray(np.array([4, 5], dtype=np.int64))
    onetags = ak._v2.index.Index8(np.array([0, 0, 0, 0, 1, 1], dtype=np.int8))
    oneindex = ak._v2.index.Index64(np.array([0, 1, 2, 3, 0, 1], dtype=np.int64))
    one = ak._v2.highlevel.Array(
        ak._v2.contents.UnionArray(onetags, oneindex, [one0, one1])
    )

    two0 = ak._v2.contents.NumpyArray(np.array([0, 100], dtype=np.int64))
    two1 = ak._v2.contents.NumpyArray(
        np.array([200.3, 300.3, 400.4, 500.5], dtype=np.float64)
    )
    twotags = ak._v2.index.Index8(np.array([0, 0, 1, 1, 1, 1], dtype=np.int8))
    twoindex = ak._v2.index.Index64(np.array([0, 1, 0, 1, 2, 3], dtype=np.int64))
    two = ak._v2.highlevel.Array(
        ak._v2.contents.UnionArray(twotags, twoindex, [two0, two1])
    )

    assert ak.to_list(one) == [0.0, 1.1, 2.2, 3.3, 4, 5]
    assert ak.to_list(two) == [0, 100, 200.3, 300.3, 400.4, 500.5]
    assert ak.to_list(one + two) == [0.0, 101.1, 202.5, 303.6, 404.4, 505.5]
