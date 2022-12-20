# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward.types import ArrayType, ListType, NumpyType, OptionType, RegularType


def test_simple():
    a = ak.from_numpy(np.array([[1, 2], [3, 4], [5, 6]]), regulararray=True)
    b = ak.from_numpy(np.array([[7, 8], [9, 10]]), regulararray=True)
    c = a.layout._mergemany([b.layout])
    assert isinstance(c, ak.contents.RegularArray)
    assert c.size == 2
    assert ak.operations.to_list(c) == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def test_regular_regular():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2])
    assert c.to_list() == [[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 2), 5)


def test_regular_option():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], null, [8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2])
    assert c.to_list() == [
        [0.0, 1.1],
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        None,
        [8.8, 9.9],
    ]
    assert c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 6)


def test_option_regular():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2])
    assert c.to_list() == [
        [0.0, 1.1],
        None,
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        [8.8, 9.9],
    ]
    assert c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 6)


def test_option_option():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], null, [8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2])
    assert c.to_list() == [
        [0.0, 1.1],
        None,
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        None,
        [8.8, 9.9],
    ]
    assert c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 7)


def test_regular_numpy():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.Array(np.array([[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]))
    a1 = ak.to_regular(a1, axis=1)
    assert isinstance(a2.layout, ak.contents.NumpyArray)
    c = ak.concatenate([a1, a2])
    assert c.to_list() == [[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 2), 5)


def test_numpy_regular():
    a1 = ak.Array(np.array([[0.0, 1.1], [2.2, 3.3]]))
    a2 = ak.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    assert isinstance(a1.layout, ak.contents.NumpyArray)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2])
    assert c.to_list() == [[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 2), 5)


def test_regular_regular_axis1():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 5), 2)


def test_option_regular_axis1():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], [7, 8, 9], [7.7, 8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2], axis=1)
    assert c.to_list() == [
        [0.0, 1.1, 4.4, 5.5, 6.6],
        [7, 8, 9],
        [2.2, 3.3, 7.7, 8.8, 9.9],
    ]
    assert c.type == ArrayType(ListType(NumpyType("float64")), 3)


def test_regular_option_axis1():
    a1 = ak.from_json("[[0.0, 1.1], [7, 8], [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], null, [7.7, 8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [7, 8], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(ListType(NumpyType("float64")), 3)


def test_option_option_axis1():
    a1 = ak.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak.from_json("[[4.4, 5.5, 6.6], null, [7.7, 8.8, 9.9]]")
    a1 = ak.to_regular(a1, axis=1)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(ListType(NumpyType("float64")), 3)


def test_regular_numpy_axis1():
    a1 = ak.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak.Array(np.array([[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]))
    a1 = ak.to_regular(a1, axis=1)
    assert isinstance(a2.layout, ak.contents.NumpyArray)
    c = ak.concatenate([a1, a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 5), 2)


def test_numpy_regular_axis1():
    a1 = ak.Array(np.array([[0.0, 1.1], [2.2, 3.3]]))
    a2 = ak.from_json("[[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]")
    assert isinstance(a1.layout, ak.contents.NumpyArray)
    a2 = ak.to_regular(a2, axis=1)
    c = ak.concatenate([a1, a2], axis=1)
    assert c.to_list() == [[0.0, 1.1, 4.4, 5.5, 6.6], [2.2, 3.3, 7.7, 8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 5), 2)
