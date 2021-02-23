# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_first_issue():
    a = ak.layout.NumpyArray(np.arange(122))
    idx = ak.layout.Index64([0, 2, 4, 6, 8, 10, 12])
    a = ak.layout.ListOffsetArray64(idx, a)
    idx = ak.layout.Index64([0, -1, 1, 2, -1, 3, 4, 5])
    a = ak.layout.IndexedOptionArray64(idx, a)
    a = ak.Array(a)
    with pytest.raises(ValueError):
        a[[[0], None]]
    assert a[[[0], None, [], [], [], [], [], []]].tolist() == [
        [0],
        None,
        [],
        [],
        None,
        [],
        [],
        [],
    ]


def test_second_issue():
    a = ak.layout.NumpyArray(np.arange(122))
    idx = ak.layout.Index64([0, 2, 4, 6, 8, 10, 12])
    a = ak.layout.ListOffsetArray64(idx, a)
    idx = ak.layout.Index64([0, -1, 1, 2, -1, 3, 4, 5])
    a = ak.layout.IndexedOptionArray64(idx, a)
    a = ak.Array(a)
    assert ak.is_valid(a)

    assert ak.is_valid(ak.argsort(a))
    assert a[ak.argsort(a)].tolist() == [
        [0, 1],
        None,
        [2, 3],
        [4, 5],
        None,
        [6, 7],
        [8, 9],
        [10, 11],
    ]
