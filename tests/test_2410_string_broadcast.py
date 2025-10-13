# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_deep_string_string():
    _left, right = ak.broadcast_arrays(
        [["x", "yz"], ["hello", "world", "foo", "bar"]], ["x", "y"]
    )
    assert right.to_list() == [["x", "x"], ["y", "y", "y", "y"]]


def test_deep_numbers_string():
    _left, right = ak.broadcast_arrays([[1, 2], [3, 4, 5, 6]], ["x", "y"])
    assert right.to_list() == [["x", "x"], ["y", "y", "y", "y"]]


def test_deep_string_numbers():
    _left, right = ak.broadcast_arrays(
        [["x", "yz"], ["hello", "world", "foo", "bar"]], [1, 2]
    )
    assert right.to_list() == [[1, 1], [2, 2, 2, 2]]


def test_same_depth():
    left, right = ak.broadcast_arrays(["z"], ["x", "y"])
    assert left.to_list() == ["z", "z"]
    assert right.to_list() == ["x", "y"]
