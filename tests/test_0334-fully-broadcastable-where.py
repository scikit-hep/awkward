# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    one = ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    two = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
    condition = ak.Array(
        [[False, True, False], [], [True, False], [True], [False, False, True, True]]
    )
    assert ak.where(condition, one, two).tolist() == [
        [0, 1, 2.2],
        [],
        [3, 4.4],
        [5],
        [6.6, 7.7, 8, 9],
    ]


def test_issue_334():
    a = ak.Array([1, 2, 3, 4])
    b = ak.Array([-1])
    c = ak.Array([True, False, True, True])

    assert ak.where(c, a, b).tolist() == [1, -1, 3, 4]
    assert ak.where(*ak.broadcast_arrays(c, a, b)).tolist() == [1, -1, 3, 4]
    assert ak.where(c, a, -1).tolist() == [1, -1, 3, 4]
    assert ak.where(*ak.broadcast_arrays(c, a, -1)).tolist() == [1, -1, 3, 4]
