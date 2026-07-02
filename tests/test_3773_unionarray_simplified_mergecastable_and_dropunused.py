# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_mergecastable():
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 1, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, mergecastable="same_kind"
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1]),
        ak.index.Index64([0, 4, 6, 1]),
        [
            ak.contents.NumpyArray(
                np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64)
            ),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 1, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, mergecastable="same_kind"
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1]),
        ak.index.Index64([0, 4, 6, 1]),
        [
            ak.contents.NumpyArray(
                np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64)
            ),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 1, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, mergecastable="equiv"
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 1, 2]),
        ak.index.Index64([0, 1, 3, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 1, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, mergecastable="same_kind"
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1]),
        ak.index.Index64([0, 4, 6, 1]),
        [
            ak.contents.NumpyArray(
                np.array(
                    [1.0, 2.0, 3.0, 1.1, 2.2, 3.3, 1.0, 2.0, 3.0], dtype=np.float64
                )
            ),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 1, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, mergecastable="equiv"
    )
    assert simplified.is_equal_to(array)


def test_dropunused():
    array = ak.contents.UnionArray(
        ak.index.Index8([2, 2, 1, 3, 3]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 0, 2, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 1, 2, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 1, 0, 0]),
        ak.index.Index64([0, 1, 0, 3, 4]),
        [
            ak.contents.NumpyArray(
                np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=np.float64)
            ),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0, 2, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.NumpyArray(
        np.array([1.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float64)
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1, 1, 1]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.NumpyArray(np.array([0, 1, 0, 0, 1], dtype=np.bool_))
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([2, 3, 2, 3, 2]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 3, 0, 3, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1, 0]),
        ak.index.Index64([0, 1, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.dtype("M8[s]"))),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([]),
        [
            ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([0, 1, 0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    assert simplified.is_equal_to(array)

    array = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0]),
        ak.index.Index64([1, 0, 2]),
        [
            ak.contents.NumpyArray(np.array([5, 10, 15], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([99], dtype=np.float64)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.NumpyArray(np.array([10, 5, 15], dtype=np.int64))
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([1, 0, 2]),
        [
            ak.contents.NumpyArray(np.array([99], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([5, 10, 15], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([99], dtype=np.float64)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, dropunused=True
    )
    expected = ak.contents.NumpyArray(np.array([10, 5, 15], dtype=np.int64))
    assert simplified.is_equal_to(expected)


def test_mergecastable_and_dropunused():
    array = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1, 3]),
        ak.index.Index64([0, 0, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([999.9], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags,
        array.index,
        array.contents,
        mergecastable="same_kind",
        dropunused=True,
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1]),
        ak.index.Index64([0, 2, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3, 4], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1, 3]),
        ak.index.Index64([0, 0, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([999.9], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags,
        array.index,
        array.contents,
        mergecastable="same_kind",
        dropunused=False,
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0, 1]),
        ak.index.Index64([1, 3, 2, 0]),
        [
            ak.contents.NumpyArray(
                np.array([999.9, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            ),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1, 3]),
        ak.index.Index64([0, 0, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([999.9], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, mergecastable="equiv", dropunused=True
    )
    expected = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 2]),
        ak.index.Index64([0, 0, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.bool_)),
        ],
    )
    assert simplified.is_equal_to(expected)

    array = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1, 3]),
        ak.index.Index64([0, 0, 1, 0]),
        [
            ak.contents.NumpyArray(np.array([999.9], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 1], dtype=np.bool_)),
        ],
    )
    simplified = array.simplified(
        array.tags, array.index, array.contents, mergecastable="equiv", dropunused=False
    )
    assert simplified.is_equal_to(array)
