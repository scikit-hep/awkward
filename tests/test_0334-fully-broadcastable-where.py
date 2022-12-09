# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    one = ak.highlevel.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    two = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    )
    condition = ak.highlevel.Array(
        [[False, True, False], [], [True, False], [True], [False, False, True, True]]
    )
    assert ak.operations.where(condition, one, two).to_list() == [
        [0, 1, 2.2],
        [],
        [3, 4.4],
        [5],
        [6.6, 7.7, 8, 9],
    ]


def test_issue_334():
    a = ak.highlevel.Array([1, 2, 3, 4])
    b = ak.highlevel.Array([-1])
    c = ak.highlevel.Array([True, False, True, True])

    assert ak.operations.where(c, a, b).to_list() == [1, -1, 3, 4]
    assert ak.operations.where(*ak.operations.broadcast_arrays(c, a, b)).to_list() == [
        1,
        -1,
        3,
        4,
    ]
    assert ak.operations.where(c, a, -1).to_list() == [1, -1, 3, 4]
    assert ak.operations.where(*ak.operations.broadcast_arrays(c, a, -1)).to_list() == [
        1,
        -1,
        3,
        4,
    ]
