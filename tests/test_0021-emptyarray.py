# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os
import json

import pytest
import numpy

import awkward1

def test_unknown():
    a = awkward1.from_json("[[], [], []]", highlevel=False)
    assert awkward1.to_list(a) == [[], [], []]
    assert str(awkward1.type(a)) == "var * unknown"
    assert awkward1.type(a) == awkward1.types.ListType(awkward1.types.UnknownType())
    assert not awkward1.type(a) == awkward1.types.PrimitiveType("float64")

    a = awkward1.from_json("[[], [[], []], [[], [], []]]", highlevel=False)
    assert awkward1.to_list(a) == [[], [[], []], [[], [], []]]
    assert str(awkward1.type(a)) == "var * var * unknown"
    assert awkward1.type(a) == awkward1.types.ListType(awkward1.types.ListType(awkward1.types.UnknownType()))

    a = awkward1.layout.ArrayBuilder()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    assert awkward1.to_list(a) == [[], [], []]
    assert str(awkward1.type(a)) == "var * unknown"
    assert awkward1.type(a) == awkward1.types.ListType(awkward1.types.UnknownType())
    assert not awkward1.type(a) == awkward1.types.PrimitiveType("float64")

    a = a.snapshot()
    assert awkward1.to_list(a) == [[], [], []]
    assert str(awkward1.type(a)) == "var * unknown"
    assert awkward1.type(a) == awkward1.types.ListType(awkward1.types.UnknownType())
    assert not awkward1.type(a) == awkward1.types.PrimitiveType("float64")

def test_getitem():
    a = awkward1.from_json("[]")
    a = awkward1.from_json("[[], [[], []], [[], [], []]]")
    assert awkward1.to_list(a[2]) == [[], [], []]

    assert awkward1.to_list(a[2, 1]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, 0]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
    assert awkward1.to_list(a[2, 1][()]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][0]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
    assert awkward1.to_list(a[2, 1][100:200]) == []
    assert awkward1.to_list(a[2, 1, 100:200]) == []
    assert awkward1.to_list(a[2, 1][numpy.array([], dtype=int)]) == []
    assert awkward1.to_list(a[2, 1, numpy.array([], dtype=int)]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, numpy.array([0], dtype=int)]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 0]
    assert ", too many dimensions in slice" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 200:300]
    assert ", too many dimensions in slice" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, numpy.array([], dtype=int)]
    assert ", too many dimensions in slice" in str(excinfo.value)

    assert awkward1.to_list(a[1:, 1:]) == [[[]], [[], []]]
    with pytest.raises(ValueError) as excinfo:
        a[1:, 1:, 0]
    assert " attempting to get 0, index out of range" in str(excinfo.value)
