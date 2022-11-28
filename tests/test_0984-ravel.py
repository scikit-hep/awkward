# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list

content = ak.contents.NumpyArray(np.array([0, 1, 2, 3, 4, 3, 6, 5, 2, 2]))


def test_one_level():
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 6, 6, 8, 10], dtype=np.int64)), content
    )

    # Test that all one level of nesting is removed
    assert to_list(ak.operations.ravel(layout)) == [
        0,
        1,
        2,
        3,
        4,
        3,
        6,
        5,
        2,
        2,
    ]


def test_two_levels():
    inner = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 6, 6, 8, 10], dtype=np.int64)), content
    )
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 4, 5], dtype=np.int64)), inner
    )

    # Test that all one level of nesting is removed
    assert to_list(ak.operations.ravel(layout)) == [
        0,
        1,
        2,
        3,
        4,
        3,
        6,
        5,
        2,
        2,
    ]


def test_record():
    x = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 5], dtype=np.int64)), content
    )
    y = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([5, 7, 10], dtype=np.int64)), content
    )
    layout = ak.contents.RecordArray((x, y), ("x", "y"))
    assert to_list(ak.operations.ravel(layout)) == [
        0,
        1,
        2,
        3,
        4,
        3,
        6,
        5,
        2,
        2,
    ]


def test_option():
    inner = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 6, 6, 8, 10], dtype=np.int64)), content
    )

    # Test that Nones are omitted
    layout = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 2, -1, 4])), inner
    )
    assert to_list(ak.operations.ravel(layout)) == [0, 1, 2, 2, 2]
