# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

content = ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4, 3, 6, 5, 2, 2]))


def test_one_level():
    layout = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 6, 6, 8, 10], dtype=np.int64)), content
    )

    # Test that all one level of nesting is removed
    assert ak.to_list(ak.ravel(layout)) == [0, 1, 2, 3, 4, 3, 6, 5, 2, 2]


def test_two_levels():
    inner = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 6, 6, 8, 10], dtype=np.int64)), content
    )
    layout = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 2, 4, 5], dtype=np.int64)), inner
    )

    # Test that all one level of nesting is removed
    assert ak.to_list(ak.ravel(layout)) == [0, 1, 2, 3, 4, 3, 6, 5, 2, 2]


def test_record():
    x = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 5], dtype=np.int64)), content
    )
    y = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([5, 7, 10], dtype=np.int64)), content
    )
    layout = ak.layout.RecordArray((x, y), ("x", "y"))
    assert ak.to_list(ak.ravel(layout)) == [0, 1, 2, 3, 4, 3, 6, 5, 2, 2]


def test_option():
    inner = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 6, 6, 8, 10], dtype=np.int64)), content
    )

    # Test that Nones are omitted
    layout = ak.layout.IndexedOptionArray64(
        ak.layout.Index64(np.array([0, -1, 2, -1, 4])), inner
    )
    assert ak.to_list(ak.ravel(layout)) == [0, 1, 2, 2, 2]
