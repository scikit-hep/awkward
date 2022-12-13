# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_trailing_sublist():
    layout = ak.contents.ListOffsetArray(
        ak.index.Index(np.array([0, 6, 9, 9], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([1, 1, 2, 2, 2, 3, 4, 4, 5])),
    )
    assert ak.run_lengths(layout).to_list() == [[2, 3, 1], [2, 1], []]


def test_leading_sublist():
    layout = ak.contents.ListOffsetArray(
        ak.index.Index(np.array([0, 0, 6, 9], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([1, 1, 2, 2, 2, 3, 4, 4, 5])),
    )
    assert ak.run_lengths(layout).to_list() == [[], [2, 3, 1], [2, 1]]


def test_leading_trailing_sublist():
    layout = ak.contents.ListOffsetArray(
        ak.index.Index(np.array([0, 0, 6, 9, 9], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([1, 1, 2, 2, 2, 3, 4, 4, 5])),
    )
    assert ak.run_lengths(layout).to_list() == [[], [2, 3, 1], [2, 1], []]
