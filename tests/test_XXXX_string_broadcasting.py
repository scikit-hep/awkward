# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_string_list_left_broadcast():
    result = ak.broadcast_arrays([["hi", "bye"], ["this", "that", "the"]], [2, 3])
    assert result[0].tolist() == [["hi", "bye"], ["this", "that", "the"]]
    assert result[1].tolist() == [[2, 2], [3, 3, 3]]


def test_string_left_broadcast_list():
    result = ak.broadcast_arrays(["hi", "bye", "this"], [[1, 2, 3], [4], []])
    assert result[0].tolist() == [["hi", "hi", "hi"], ["bye"], []]
    assert result[1].tolist() == [[1, 2, 3], [4], []]


def test_string_list():
    result = ak.broadcast_arrays(["hi", "bye", "this"], [1, 2, 3])
    assert result[0].tolist() == ["hi", "bye", "this"]
    assert result[1].tolist() == [1, 2, 3]
