# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_first_issue():
    a = ak.contents.NumpyArray(np.arange(122))
    idx = ak.index.Index64([0, 2, 4, 6, 8, 10, 12])
    a = ak.contents.ListOffsetArray(idx, a)
    idx = ak.index.Index64([0, -1, 1, 2, -1, 3, 4, 5])
    a = ak.contents.IndexedOptionArray(idx, a)
    a = ak.Array(a)
    with pytest.raises(IndexError):
        a[[[0], None]]
    assert a[[[0], None, [], [], [], [], [], []]].to_list() == [
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
    a = ak.contents.NumpyArray(np.arange(122))
    idx = ak.index.Index64([0, 2, 4, 6, 8, 10, 12])
    a = ak.contents.ListOffsetArray(idx, a)
    idx = ak.index.Index64([0, -1, 1, 2, -1, 3, 4, 5])
    a = ak.contents.IndexedOptionArray(idx, a)
    a = ak.Array(a)
    assert ak.operations.is_valid(a)

    assert ak.operations.is_valid(ak.operations.argsort(a))
    assert a[ak.operations.argsort(a)].to_list() == [
        [0, 1],
        None,
        [2, 3],
        [4, 5],
        None,
        [6, 7],
        [8, 9],
        [10, 11],
    ]
