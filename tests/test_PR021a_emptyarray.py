# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os
import json

import pytest
import numpy

import awkward1

def test_unknown():
    a = awkward1.fromjson("[[], [], []]").layout
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "var * unknown"
    assert awkward1.typeof(a) == awkward1.types.ListType(awkward1.types.UnknownType())
    assert not awkward1.typeof(a) == awkward1.types.PrimitiveType("float64")

    a = awkward1.fromjson("[[], [[], []], [[], [], []]]").layout
    assert awkward1.tolist(a) == [[], [[], []], [[], [], []]]
    assert str(awkward1.typeof(a)) == "var * var * unknown"
    assert awkward1.typeof(a) == awkward1.types.ListType(awkward1.types.ListType(awkward1.types.UnknownType()))

    a = awkward1.layout.FillableArray()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "var * unknown"
    assert awkward1.typeof(a) == awkward1.types.ListType(awkward1.types.UnknownType())
    assert not awkward1.typeof(a) == awkward1.types.PrimitiveType("float64")

    a = a.snapshot()
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "var * unknown"
    assert awkward1.typeof(a) == awkward1.types.ListType(awkward1.types.UnknownType())
    assert not awkward1.typeof(a) == awkward1.types.PrimitiveType("float64")

def test_getitem():
    a = awkward1.fromjson("[[], [[], []], [[], [], []]]")
    assert awkward1.tolist(a[2]) == [[], [], []]

    assert awkward1.tolist(a[2, 1]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, 0]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"
    assert awkward1.tolist(a[2, 1][()]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][0]
    assert str(excinfo.value) == "in EmptyArray attempting to get 0, index out of range"
    assert awkward1.tolist(a[2, 1][100:200]) == []
    assert awkward1.tolist(a[2, 1, 100:200]) == []
    assert awkward1.tolist(a[2, 1][numpy.array([], dtype=int)]) == []
    assert awkward1.tolist(a[2, 1, numpy.array([], dtype=int)]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, numpy.array([0], dtype=int)]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 0]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, 200:300]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"
    with pytest.raises(ValueError) as excinfo:
        a[2, 1][100:200, numpy.array([], dtype=int)]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"

    assert awkward1.tolist(a[1:, 1:]) == [[[]], [[], []]]
    with pytest.raises(ValueError) as excinfo:
        a[1:, 1:, 0]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"
