# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test():
    one = ak._v2.highlevel.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    two = ak._v2.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    )
    condition = ak._v2.highlevel.Array(
        [[False, True, False], [], [True, False], [True], [False, False, True, True]]
    )
    assert ak._v2.operations.where(condition, one, two).tolist() == [
        [0, 1, 2.2],
        [],
        [3, 4.4],
        [5],
        [6.6, 7.7, 8, 9],
    ]


def test_issue_334():
    a = ak._v2.highlevel.Array([1, 2, 3, 4])
    b = ak._v2.highlevel.Array([-1])
    c = ak._v2.highlevel.Array([True, False, True, True])

    assert ak._v2.operations.where(c, a, b).tolist() == [1, -1, 3, 4]
    assert ak._v2.operations.where(
        *ak._v2.operations.broadcast_arrays(c, a, b)
    ).tolist() == [1, -1, 3, 4]
    assert ak._v2.operations.where(c, a, -1).tolist() == [1, -1, 3, 4]
    assert ak._v2.operations.where(
        *ak._v2.operations.broadcast_arrays(c, a, -1)
    ).tolist() == [1, -1, 3, 4]
