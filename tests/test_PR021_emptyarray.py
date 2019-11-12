# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os
import json

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1

def test_unknown():
    a = awkward1.fromjson("[[], [], []]")
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "3 * var * ???"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(3, awkward1.layout.ListType(awkward1.layout.UnknownType()))
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(3, awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64")))
    assert awkward1.typeof(a) != awkward1.layout.ArrayType(3, awkward1.layout.PrimitiveType("float64"))

    a = awkward1.fromjson("[[], [[], []], [[], [], []]]")
    assert awkward1.tolist(a) == [[], [[], []], [[], [], []]]
    assert str(awkward1.typeof(a)) == "3 * var * var * ???"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(3, awkward1.layout.ListType(awkward1.layout.ListType(awkward1.layout.UnknownType())))

    a = awkward1.layout.FillableArray()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "3 * var * ???"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(3, awkward1.layout.ListType(awkward1.layout.UnknownType()))
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(3, awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64")))
    assert awkward1.typeof(a) != awkward1.layout.ArrayType(3, awkward1.layout.PrimitiveType("float64"))

    a = a.snapshot()
    assert awkward1.tolist(a) == [[], [], []]
    assert str(awkward1.typeof(a)) == "3 * var * ???"
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(3, awkward1.layout.ListType(awkward1.layout.UnknownType()))
    assert awkward1.typeof(a) == awkward1.layout.ArrayType(3, awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64")))
    assert awkward1.typeof(a) != awkward1.layout.ArrayType(3, awkward1.layout.PrimitiveType("float64"))

def test_getitem():
    a = awkward1.fromjson("[[], [[], []], [[], [], []]]")
    assert awkward1.tolist(a[2]) == [[], [], []]
    assert awkward1.tolist(a[2, 1]) == []
    with pytest.raises(ValueError) as excinfo:
        a[2, 1, 0]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"

    assert awkward1.tolist(a[1:, 1:]) == [[[]], [[], []]]
    with pytest.raises(ValueError) as excinfo:
        a[1:, 1:, 0]
    assert str(excinfo.value) == "in ListArray64 attempting to get 0, index out of range"
    with pytest.raises(ValueError) as excinfo:
        a[1:, 1:, 1:]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"

    with pytest.raises(ValueError) as excinfo:
        a[1:, 1:, numpy.array([], dtype=int)]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"
    with pytest.raises(ValueError) as excinfo:
        a[1:, numpy.array([], dtype=int), numpy.array([], dtype=int)]
    assert str(excinfo.value) == "in EmptyArray, too many dimensions in slice"
