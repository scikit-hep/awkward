# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test_reduce_1d_strided():
    non_contiguous_array = np.arange(64, dtype=np.int64)[::3]
    layout = ak.contents.NumpyArray(
        non_contiguous_array,
    )
    assert not layout.is_contiguous
    assert ak.sum(layout, axis=-1) == np.sum(non_contiguous_array, axis=-1)


def test_reduce_transpose_2d():
    non_contiguous_array = np.arange(6 * 8, dtype=np.int64).reshape(6, 8).T
    layout = ak.contents.NumpyArray(
        non_contiguous_array,
    )
    assert not layout.is_contiguous
    assert (
        ak.sum(layout, axis=-1).to_list()
        == np.sum(non_contiguous_array, axis=-1).tolist()
    )


def test_reduce_2d():
    non_contiguous_array = np.arange(6 * 8, dtype=np.int64).reshape(6, 8)
    layout = ak.contents.NumpyArray(
        non_contiguous_array,
    )
    assert layout.is_contiguous
    assert (
        ak.sum(layout, axis=-1).to_list()
        == np.sum(non_contiguous_array, axis=-1).tolist()
    )


def test_unique_1d_strided():
    non_contiguous_array = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)[::2]
    layout = ak.contents.NumpyArray(
        non_contiguous_array,
    )
    assert not layout.is_contiguous
    assert layout.unique(-1).to_list() == [0, 1, 2, 3]
    assert layout.is_unique(-1)

    non_contiguous_array = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)[::2]
    layout = ak.contents.NumpyArray(
        non_contiguous_array,
    )
    assert not layout.is_contiguous
    assert layout.unique(-1).to_list() == [0, 1, 2, 3]
    assert not layout.is_unique(-1)


def test_unique_2d_strided():
    transposed_array = (
        np.array([0, 0, 1, 0, 2, 2, 3, 2, 3, 3, 4, 4, 4, 5, 6], dtype=np.int64)
        .reshape(-1, 3)
        .T
    )
    layout = ak.contents.NumpyArray(transposed_array)
    assert not layout.is_contiguous
    assert layout.unique(-1).to_list() == [[0, 3, 4], [0, 2, 4, 5], [1, 2, 3, 4, 6]]
    assert not layout.is_unique(-1)
