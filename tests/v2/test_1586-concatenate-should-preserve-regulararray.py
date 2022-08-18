# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.types import ArrayType, RegularType, OptionType, NumpyType


def test_simple():
    a = ak._v2.from_numpy(np.array([[1, 2], [3, 4], [5, 6]]), regulararray=True)
    b = ak._v2.from_numpy(np.array([[7, 8], [9, 10]]), regulararray=True)
    c = a.layout.merge(b.layout)
    assert isinstance(c, ak._v2.contents.RegularArray)
    assert c.size == 2
    assert ak._v2.operations.to_list(c) == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def test_regular_regular():
    a1 = ak._v2.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak._v2.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    a1 = ak._v2.to_regular(a1, axis=1)
    a2 = ak._v2.to_regular(a2, axis=1)
    c = ak._v2.concatenate([a1, a2])
    assert c.tolist() == [[0.0, 1.1], [2.2, 3.3], [4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]
    assert c.type == ArrayType(RegularType(NumpyType("float64"), 2), 5)


def test_regular_option():
    a1 = ak._v2.from_json("[[0.0, 1.1], [2.2, 3.3]]")
    a2 = ak._v2.from_json("[[4.4, 5.5], [6.6, 7.7], null, [8.8, 9.9]]")
    a1 = ak._v2.to_regular(a1, axis=1)
    a2 = ak._v2.to_regular(a2, axis=1)
    c = ak._v2.concatenate([a1, a2])
    assert c.tolist() == [
        [0.0, 1.1],
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        None,
        [8.8, 9.9],
    ]
    assert c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 6)


def test_option_regular():
    a1 = ak._v2.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak._v2.from_json("[[4.4, 5.5], [6.6, 7.7], [8.8, 9.9]]")
    a1 = ak._v2.to_regular(a1, axis=1)
    a2 = ak._v2.to_regular(a2, axis=1)
    c = ak._v2.concatenate([a1, a2])
    assert c.tolist() == [
        [0.0, 1.1],
        None,
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        [8.8, 9.9],
    ]
    assert c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 6)


def test_option_option():
    a1 = ak._v2.from_json("[[0.0, 1.1], null, [2.2, 3.3]]")
    a2 = ak._v2.from_json("[[4.4, 5.5], [6.6, 7.7], null, [8.8, 9.9]]")
    a1 = ak._v2.to_regular(a1, axis=1)
    a2 = ak._v2.to_regular(a2, axis=1)
    c = ak._v2.concatenate([a1, a2])
    assert c.tolist() == [
        [0.0, 1.1],
        None,
        [2.2, 3.3],
        [4.4, 5.5],
        [6.6, 7.7],
        None,
        [8.8, 9.9],
    ]
    assert c.type == ArrayType(OptionType(RegularType(NumpyType("float64"), 2)), 7)
