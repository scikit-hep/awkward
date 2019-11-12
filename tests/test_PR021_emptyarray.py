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
