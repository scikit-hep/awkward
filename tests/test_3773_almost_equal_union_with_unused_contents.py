# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak

# Equal cases - all using zeros


def test_equal_empty_union():
    array = ak.contents.UnionArray(
        ak.index.Index8([]),
        ak.index.Index64([]),
        [
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_size_one_tags():
    array = ak.contents.UnionArray(
        ak.index.Index8([0]),
        ak.index.Index64([0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)

    array = ak.contents.UnionArray(
        ak.index.Index8([1]),
        ak.index.Index64([0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_tags_not_starting_at_zero():
    array = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_single_tag_value():
    array = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([0, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_sparse_tags():
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0, 2]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_different_tag_ordering():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_tags_high_values():
    array = ak.contents.UnionArray(
        ak.index.Index8([3, 4, 5, 3]),
        ak.index.Index64([0, 0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_tags_large_gap():
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 10, 0, 10]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_middle_contents_only():
    left = ak.contents.UnionArray(
        ak.index.Index8([2, 3, 2]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_last_content_unused():
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_first_content_unused():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_multiple_trailing_unused():
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


def test_equal_different_unique_tag_counts_left_more():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_different_unique_tag_counts_right_more():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([0, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0, 0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2]),
        ak.index.Index64([0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_different_unique_tag_counts_complex():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1, 1]),
        ak.index.Index64([0, 1, 2, 3]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0, 0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_all_tags_use_same_content():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0]),
        ak.index.Index64([0, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([0, 0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([0, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_alternating_pattern():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 0, 1, 0, 1]),
        ak.index.Index64([0, 0, 0, 0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 0, 1, 0, 1, 0]),
        ak.index.Index64([0, 0, 0, 0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_all_unique_tags():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3, 4]),
        ak.index.Index64([0, 0, 0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([5, 6, 7, 8, 9]),
        ak.index.Index64([0, 0, 0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_single_element():
    left = ak.contents.UnionArray(
        ak.index.Index8([5]),
        ak.index.Index64([0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([7]),
        ak.index.Index64([0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_repeated_indices():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 0]),
        ak.index.Index64([0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(left, right)
    assert ak.almost_equal(left, right)


def test_equal_indices_at_end_of_content():
    array = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([4, 2]),
        [
            ak.contents.NumpyArray(np.array([0, 0, 0, 0, 0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0, 0, 0], dtype=np.int64)),
        ],
    )
    assert ak.array_equal(array, array)
    assert ak.almost_equal(array, array)


# Not equal cases - using different values


def test_not_equal_same_tags_different_values():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_nonzero_tags():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 5], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_sparse_tags():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0, 2]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 2, 0, 2]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 99], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_high_tags():
    left = ak.contents.UnionArray(
        ak.index.Index8([2, 5, 2, 5]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([2, 5, 2, 5]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 99], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_different_lengths():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([0, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_last_content_unused():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 3, 0, 3]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 4], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 3, 0, 3]),
        ak.index.Index64([0, 0, 1, 1]),
        [
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3, 99], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_first_content_unused():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 2, 1]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([99], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_middle_contents_only():
    left = ak.contents.UnionArray(
        ak.index.Index8([2, 3, 2]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([2, 3, 2]),
        ak.index.Index64([0, 0, 1]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 99], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_different_unique_tag_counts_left_more():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 2]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([5], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([10], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1]),
        ak.index.Index64([0, 0]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_different_unique_tag_counts_right_more():
    left = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1]),
        ak.index.Index64([0, 1, 2]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2]),
        ak.index.Index64([0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([99], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)


def test_not_equal_different_unique_tag_counts_complex():
    left = ak.contents.UnionArray(
        ak.index.Index8([0, 1, 2, 3]),
        ak.index.Index64([0, 0, 0, 0]),
        [
            ak.contents.NumpyArray(np.array([1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([4], dtype=np.int64)),
        ],
    )
    right = ak.contents.UnionArray(
        ak.index.Index8([1, 1, 1, 1]),
        ak.index.Index64([0, 1, 2, 3]),
        [
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([1, 2, 3, 999], dtype=np.int64)),
        ],
    )
    assert not ak.array_equal(left, right)
    assert not ak.almost_equal(left, right)
