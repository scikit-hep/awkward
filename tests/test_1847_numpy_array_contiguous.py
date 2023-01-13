# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


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
