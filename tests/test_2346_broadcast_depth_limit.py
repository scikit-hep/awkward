# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak


def test():
    left = ak.Array([0, 4, 5])
    right = ak.Array([[0, 1, 2, 3], [5, 6, 7, 2], [9, 8, 2, 0]])

    left_result, right_result = ak.broadcast_arrays(left, right, depth_limit=4)
    assert left_result.to_list() == [[0, 0, 0, 0], [4, 4, 4, 4], [5, 5, 5, 5]]
    assert right_result.to_list() == [[0, 1, 2, 3], [5, 6, 7, 2], [9, 8, 2, 0]]
