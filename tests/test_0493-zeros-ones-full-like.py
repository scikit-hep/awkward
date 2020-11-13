# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    array = awkward1.Array([[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}], [], [{"x": 3.3, "y": [1, 2, None, 3]}, False, False, True, {"x": 4.4, "y": [1, 2, None, 3, 4]}]])

    assert awkward1.full_like(array, 12.3).tolist() == [[{"x": 12.3, "y": []}, {"x": 12.3, "y": [12]}, {"x": 12.3, "y": [12, 12]}], [], [{"x": 12.3, "y": [12, 12, None, 12]}, True, True, True, {"x": 12.3, "y": [12, 12, None, 12, 12]}]]

    assert awkward1.zeros_like(array).tolist() == [[{"x": 0.0, "y": []}, {"x": 0.0, "y": [0]}, {"x": 0.0, "y": [0, 0]}], [], [{"x": 0.0, "y": [0, 0, None, 0]}, False, False, False, {"x": 0.0, "y": [0, 0, None, 0, 0]}]]

    assert awkward1.ones_like(array).tolist() == [[{"x": 1.0, "y": []}, {"x": 1.0, "y": [1]}, {"x": 1.0, "y": [1, 1]}], [], [{"x": 1.0, "y": [1, 1, None, 1]}, True, True, True, {"x": 1.0, "y": [1, 1, None, 1, 1]}]]

    array = awkward1.Array([["one", "two", "three"], [], ["four", "five"]])
    assert awkward1.full_like(array, "hello").tolist() == [["hello", "hello", "hello"], [], ["hello", "hello"]]
    assert awkward1.full_like(array, 1).tolist() == [["1", "1", "1"], [], ["1", "1"]]
    assert awkward1.full_like(array, 0).tolist() == [["0", "0", "0"], [], ["0", "0"]]
    assert awkward1.ones_like(array).tolist() == [["1", "1", "1"], [], ["1", "1"]]
    assert awkward1.zeros_like(array).tolist() == [["", "", ""], [], ["", ""]]

    array = awkward1.Array([[b"one", b"two", b"three"], [], [b"four", b"five"]])
    assert awkward1.full_like(array, b"hello").tolist() == [[b"hello", b"hello", b"hello"], [], [b"hello", b"hello"]]
    assert awkward1.full_like(array, 1).tolist() == [[b"1", b"1", b"1"], [], [b"1", b"1"]]
    assert awkward1.full_like(array, 0).tolist() == [[b"0", b"0", b"0"], [], [b"0", b"0"]]
    assert awkward1.ones_like(array).tolist() == [[b"1", b"1", b"1"], [], [b"1", b"1"]]
    assert awkward1.zeros_like(array).tolist() == [[b"", b"", b""], [], [b"", b""]]
