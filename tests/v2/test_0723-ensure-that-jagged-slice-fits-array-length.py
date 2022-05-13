# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_first_issue():
    a = ak._v2.contents.NumpyArray(np.arange(122))
    idx = ak._v2.index.Index64([0, 2, 4, 6, 8, 10, 12])
    a = ak._v2.contents.ListOffsetArray(idx, a)
    idx = ak._v2.index.Index64([0, -1, 1, 2, -1, 3, 4, 5])
    a = ak._v2.contents.IndexedOptionArray(idx, a)
    a = ak._v2.Array(a)
    with pytest.raises(IndexError):
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
    a = ak._v2.contents.NumpyArray(np.arange(122))
    idx = ak._v2.index.Index64([0, 2, 4, 6, 8, 10, 12])
    a = ak._v2.contents.ListOffsetArray(idx, a)
    idx = ak._v2.index.Index64([0, -1, 1, 2, -1, 3, 4, 5])
    a = ak._v2.contents.IndexedOptionArray(idx, a)
    a = ak._v2.Array(a)
    assert ak._v2.operations.is_valid(a)

    assert ak._v2.operations.is_valid(ak._v2.operations.argsort(a))
    assert a[ak._v2.operations.argsort(a)].tolist() == [
        [0, 1],
        None,
        [2, 3],
        [4, 5],
        None,
        [6, 7],
        [8, 9],
        [10, 11],
    ]
