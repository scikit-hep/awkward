# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import pytest
import numpy

import awkward1

def test_types():
    t0 = awkward1.types.UnknownType()
    t1 = awkward1.types.PrimitiveType("int32")
    t2 = awkward1.types.OptionType(t1)
    t3 = awkward1.types.UnionType((t1, awkward1.types.PrimitiveType("float64")))
    t4 = awkward1.types.ListType(t1)
    t4b = awkward1.types.ListType(awkward1.types.PrimitiveType("int32"))
    t5 = awkward1.types.ListType(t4)
    t6 = awkward1.types.OptionType(t4)
    assert repr(t0) == "unknown"
    assert repr(t1) == "int32"
    assert repr(t2) == "?int32"
    assert repr(t3) == "union[int32, float64]"
    assert repr(t4) == "var * int32"
    assert repr(t5) == "var * var * int32"
    assert repr(t6) == "option[var * int32]"
    assert repr(t2.type) == "int32"
    assert t3.numtypes == 2
    assert repr(t3.type(0)) == "int32"
    assert repr(t3.type(1)) == "float64"
    assert [repr(x) for x in t3.types] == ["int32", "float64"]
    assert repr(t4.type) == "int32"
    assert repr(t5.type) == "var * int32"

def test_boolean():
    a = awkward1.layout.ArrayBuilder()
    a.boolean(True)
    a.boolean(True)
    a.boolean(False)
    a.boolean(True)
    assert awkward1.to_list(a.snapshot()) == [True, True, False, True]
    assert awkward1.to_list(a) == [True, True, False, True]
    assert awkward1.to_list(a[1:-1]) == [True, False]

def test_big():
    a = awkward1.layout.ArrayBuilder(initial=90)
    for i in range(2000):
        if i == 200:
            tmp = a.snapshot()
        a.boolean(i % 2 == 0)
    assert awkward1.to_list(a) == [True, False] * 1000
    assert awkward1.to_list(tmp) == [True, False] * 100

def test_integer():
    a = awkward1.layout.ArrayBuilder()
    a.integer(10)
    a.integer(9)
    a.integer(8)
    a.integer(7)
    a.integer(6)
    assert awkward1.to_list(a.snapshot()) == [10, 9, 8, 7, 6]
    assert awkward1.to_list(a) == [10, 9, 8, 7, 6]
    assert awkward1.to_list(a[1:-1]) == [9, 8, 7]

def test_real():
    a = awkward1.layout.ArrayBuilder()
    a.real(1.1)
    a.real(2.2)
    a.real(3.3)
    a.real(4.4)
    a.real(5.5)
    assert awkward1.to_list(a.snapshot()) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(a) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(a[1:-1]) == [2.2, 3.3, 4.4]

def test_integer_real():
    a = awkward1.layout.ArrayBuilder()
    a.integer(1)
    a.integer(2)
    a.real(3.3)
    a.integer(4)
    a.integer(5)
    assert awkward1.to_list(a.snapshot()) == [1.0, 2.0, 3.3, 4.0, 5.0]
    assert awkward1.to_list(a) == [1.0, 2.0, 3.3, 4.0, 5.0]
    assert awkward1.to_list(a[1:-1]) == [2.0, 3.3, 4.0]

def test_real_integer():
    a = awkward1.layout.ArrayBuilder()
    a.real(1.1)
    a.real(2.2)
    a.integer(3)
    a.real(4.4)
    a.real(5.5)
    assert awkward1.to_list(a.snapshot()) == [1.1, 2.2, 3.0, 4.4, 5.5]
    assert awkward1.to_list(a) == [1.1, 2.2, 3.0, 4.4, 5.5]
    assert awkward1.to_list(a[1:-1]) == [2.2, 3.0, 4.4]

def test_list_real():
    a = awkward1.layout.ArrayBuilder()
    a.beginlist()
    a.real(1.1)
    a.real(2.2)
    a.real(3.3)
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.real(4.4)
    a.real(5.5)
    a.endlist()
    assert awkward1.to_list(a.snapshot()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(a) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(a[1:-1]) == [[]]
    assert awkward1.to_list(a[1:]) == [[], [4.4, 5.5]]

def test_list_list_real():
    a = awkward1.layout.ArrayBuilder()
    a.beginlist()
    a.beginlist()
    a.real(1.1)
    a.real(2.2)
    a.real(3.3)
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.real(4.4)
    a.real(5.5)
    a.endlist()
    a.endlist()
    a.beginlist()
    a.endlist()
    a.beginlist()
    a.beginlist()
    a.real(6.6)
    a.real(7.7)
    a.endlist()
    a.beginlist()
    a.real(8.8)
    a.real(9.9)
    a.endlist()
    a.endlist()
    assert awkward1.to_list(a.snapshot()) == [[[1.1, 2.2, 3.3], [], [4.4, 5.5]], [], [[6.6, 7.7], [8.8, 9.9]]]
    assert awkward1.to_list(a) == [[[1.1, 2.2, 3.3], [], [4.4, 5.5]], [], [[6.6, 7.7], [8.8, 9.9]]]
    assert awkward1.to_list(a[1:]) == [[], [[6.6, 7.7], [8.8, 9.9]]]

def test_list_errors():
    with pytest.raises(ValueError):
        a = awkward1.layout.ArrayBuilder()
        a.endlist()

    with pytest.raises(ValueError):
        a = awkward1.layout.ArrayBuilder()
        a.real(3.14)
        a.endlist()

    with pytest.raises(ValueError):
        a = awkward1.layout.ArrayBuilder()
        a.beginlist()
        a.real(3.14)
        a.endlist()
        a.endlist()

    with pytest.raises(ValueError):
        a = awkward1.layout.ArrayBuilder()
        a.beginlist()
        a.beginlist()
        a.real(3.14)
        a.endlist()
        a.endlist()
        a.endlist()

    a = awkward1.layout.ArrayBuilder()
    a.beginlist()
    a.real(1.1)
    a.real(2.2)
    a.real(3.3)
    a.endlist()
    a.beginlist()
    a.real(4.4)
    a.real(5.5)
    assert awkward1.to_list(a.snapshot()) == [[1.1, 2.2, 3.3]]
    assert awkward1.to_list(a) == [[1.1, 2.2, 3.3]]
    assert awkward1.to_list(a[1:]) == []
