# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test_unknown():
    a = ak.from_json("[[], [], []]", highlevel=False)
    assert ak.to_list(a) == [[], [], []]
    assert str(ak.type(a)) == "var * unknown"
    assert ak.type(a) == ak.types.ListType(ak.types.UnknownType())
    assert not ak.type(a) == ak.types.PrimitiveType("float64")

    a = ak.from_json("[[], [[], []], [[], [], []]]", highlevel=False)
    assert ak.to_list(a) == [[], [[], []], [[], [], []]]
    assert str(ak.type(a)) == "var * var * unknown"
    assert ak.type(a) == ak.types.ListType(ak.types.ListType(ak.types.UnknownType()))

    a = ak.layout.ArrayBuilder()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    assert ak.to_list(a) == [[], [], []]
    assert str(ak.type(a)) == "var * unknown"
    assert ak.type(a) == ak.types.ListType(ak.types.UnknownType())
    assert not ak.type(a) == ak.types.PrimitiveType("float64")

    a = a.snapshot()
    assert ak.to_list(a) == [[], [], []]
    assert str(ak.type(a)) == "var * unknown"
    assert ak.type(a) == ak.types.ListType(ak.types.UnknownType())
    assert not ak.type(a) == ak.types.PrimitiveType("float64")


def test_getitem():
    a = ak.from_json("[]")
    a = ak.from_json("[[], [[], []], [[], [], []]]")
    assert ak.to_list(a[2]) == [[], [], []]

    assert ak.to_list(a[2, 1]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, 0]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
    assert ak.to_list(a[2, 1][()]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][0]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
    assert ak.to_list(a[2, 1][100:200]) == []
    assert ak.to_list(a[2, 1, 100:200]) == []
    assert ak.to_list(a[2, 1][np.array([], dtype=int)]) == []
    assert ak.to_list(a[2, 1, np.array([], dtype=int)]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, np.array([0], dtype=int)]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 0]
    assert ", too many dimensions in slice" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 200:300]
    assert ", too many dimensions in slice" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, np.array([], dtype=int)]
    assert ", too many dimensions in slice" in str(excinfo.value)

    assert ak.to_list(a[1:, 1:]) == [[[]], [[], []]]
    with pytest.raises(ValueError) as excinfo:
        a[1:, 1:, 0]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
