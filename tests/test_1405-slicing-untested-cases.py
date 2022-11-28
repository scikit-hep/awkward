# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_index_packed():
    """Base test case"""
    content = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        # Here we have a third sublist [2, 3) that isn't mapped
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 1, 2], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2, 2], dtype=np.int64)),
        ),
    )

    index = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ),
    )

    assert content[index].to_list() == [[[], [2]]]


def test_index_unmapped():
    """Check that contents with unmapped sublists still support jagged indexing"""
    content = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        # Here we have a third sublist [2, 3) that isn't mapped
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2, 2, 2], dtype=np.int64)),
        ),
    )

    index = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ),
    )

    assert content[index].to_list() == [[[], [2]]]


def test_list_option_list():
    """Check that non-offset list(option(list indexes correctly"""
    content = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([2, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2, 2, 2], dtype=np.int64)),
        ),
    )

    index = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, 1], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ),
        ),
    )

    assert content[index].to_list() == [[[], [2]]]


def test_list_option_list_offset():
    """Check that offset list(option(list indexes correctly"""
    content = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([1, 3], dtype=np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2, 2, 3], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([2, 2, 2], dtype=np.int64)),
        ),
    )

    index = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.IndexedOptionArray(
            ak.index.Index64(np.array([0, 1], dtype=np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                ak.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ),
        ),
    )

    assert content[index].to_list() == [[[], [2]]]
