# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_union_tags_equal():
    # Tags not starting at zero
    array = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)

    # Single tag value
    array = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([1, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([7, 8, 9], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)

    # Sparse tags
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0, 2]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([30], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([40, 50], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)

    # Different tag ordering, same values
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
        ],
    )
    assert not np.array_equal(left.tags.data, right.tags.data)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)

    # Different nonzero tags, same values
    left = ak.contents.UnionArray(
        ak.index.Index8([2, 5, 2, 5]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5, 3.7], dtype=np.float64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 3, 1, 3]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5, 3.7], dtype=np.float64)),
        ],
    )
    assert not np.array_equal(left.tags.data, right.tags.data)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)

    # Sparse nonzero tags, same values
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 7, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([4, 9, 4]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
        ],
    )
    assert not np.array_equal(left.tags.data, right.tags.data)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)

    # Three contents different orderings
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 0]),
        ak.index.Index64([0, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([True], dtype=np.bool_)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 0, 1]),
        ak.index.Index64([0, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([True], dtype=np.bool_)),
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
        ],
    )
    assert not np.array_equal(left.tags.data, right.tags.data)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)

    # Tags 3, 4, 5
    array = ak.contents.UnionArray(
        ak.index.Index8([3, 4, 5, 3]),
        ak.index.Index64([0, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([100, 200], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([False], dtype=np.bool_)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)

    # Max tag value 10
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 10, 0, 10]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)

    # Last content unused - tags [0, 3] with trailing unused contents
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 3, 0, 3]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.5, 2.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)

    # Last content unused - tags [1, 2] with trailing unused contents
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([5, 10], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([7.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([5, 10], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([7.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    assert not np.array_equal(left.tags.data, right.tags.data)
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)

    # Last multiple contents unused - tags [0, 2] with many trailing unused contents
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([100, 200], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3.14], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_union_tags_not_equal():
    # Different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2], dtype=np.int64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Nonzero tags, different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 5], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Sparse tags, different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0, 2]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([30], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([40, 50], dtype=np.int64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0, 2]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([30], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([40, 99], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Tags 2 and 5, different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([2, 5, 2, 5]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5, 3.7], dtype=np.float64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([2, 5, 2, 5]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([10, 21], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5, 3.7], dtype=np.float64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Nonzero tags, different lengths
    array1 = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([0, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4, 5], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Max tag 10, different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([0, 10, 0, 10]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([0, 10, 0, 10]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 9], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Different tag ordering, different values
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([2.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([10, 99], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)

    # Last content unused - tags [0, 3] with trailing unused, different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([0, 3, 0, 3]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.5, 2.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([0, 3, 0, 3]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1.5, 9.9], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Last content unused - tags [1, 2] with trailing unused, different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([5, 10], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([7.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([5, 10], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([8.5], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Last multiple contents unused - tags [0, 2] with trailing unused, different values
    array1 = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([100, 200], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3.14], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    array2 = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([100, 999], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3.14], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(array1, array2)
    assert not ak.almost_equal(array1, array2)

    # Last content unused, different tag ordering and values
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([50, 60], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([1.1], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([1.1], dtype=np.float64)),
            ak.contents.NumpyArray(np.array([50, 99], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.float64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)
