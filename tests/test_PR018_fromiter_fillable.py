# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_types():
    t1 = awkward1.layout.PrimitiveType("int32")
    t2 = awkward1.layout.OptionType(t1)
    t3 = awkward1.layout.UnionType(t1, awkward1.layout.PrimitiveType("float64"))
    assert repr(t1) == "int32"
    assert repr(t2) == "option[int32]"
    assert repr(t3) == "union[int32, float64]"
    assert repr(t2.type) == "int32"
    assert t3.numtypes == 2
    assert repr(t3.type(0)) == "int32"
    assert repr(t3.type(1)) == "float64"
    assert [repr(x) for x in t3.types] == ["int32", "float64"]

def test_boolean():
    a = awkward1.layout.FillableArray()
    a.boolean(True)
    a.boolean(True)
    a.boolean(False)
    a.boolean(True)
    assert awkward1.tolist(a.snapshot()) == [True, True, False, True]
    assert awkward1.tolist(a) == [True, True, False, True]
    assert awkward1.tolist(a[1:-1]) == [True, False]

def test_big():
    a = awkward1.layout.FillableArray(initial=90)
    for i in range(2000):
        if i == 200:
            tmp = a.snapshot()
        a.boolean(i % 2 == 0)
    assert awkward1.tolist(a) == [True, False] * 1000
    assert awkward1.tolist(tmp) == [True, False] * 100

def test_integer():
    a = awkward1.layout.FillableArray()
    a.integer(10)
    a.integer(9)
    a.integer(8)
    a.integer(7)
    a.integer(6)
    assert awkward1.tolist(a.snapshot()) == [10, 9, 8, 7, 6]
    assert awkward1.tolist(a) == [10, 9, 8, 7, 6]
    assert awkward1.tolist(a[1:-1]) == [9, 8, 7]
