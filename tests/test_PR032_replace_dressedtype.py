# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools
import pickle

import pytest
import numpy

import awkward1

def test_types_with_parameters():
    t = awkward1.layout.UnknownType()
    assert t.parameters == {}
    t.parameters = {"key": ["val", "ue"]}
    assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.UnknownType(parameters={"key": ["val", "ue"]})
    assert t.parameters == {"key": ["val", "ue"]}

    t = awkward1.layout.PrimitiveType("int32", parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.PrimitiveType("float64", parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.ArrayType(awkward1.layout.PrimitiveType("int32"), 100, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32"), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.RegularType(awkward1.layout.PrimitiveType("int32"), 5, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.OptionType(awkward1.layout.PrimitiveType("int32"), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.UnionType((awkward1.layout.PrimitiveType("int32"), awkward1.layout.PrimitiveType("float64")), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.layout.RecordType({"one": awkward1.layout.PrimitiveType("int32"), "two": awkward1.layout.PrimitiveType("float64")}, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}

    t = awkward1.layout.UnknownType(parameters={"key1": ["val", "ue"], "key2": u"one \u2192 two"})
    assert t.parameters == {"key2": u"one \u2192 two", "key1": ["val", "ue"]}

    assert t == awkward1.layout.UnknownType(parameters={"key2": u"one \u2192 two", "key1": ["val", "ue"]})
    assert t != awkward1.layout.UnknownType(parameters={"key": ["val", "ue"]})

def test_dress():
    class Dummy(awkward1.highlevel.Array):
        def __repr__(self):
            return "<Dummy {0}>".format(str(self))
    ns = {"Dummy": Dummy}

    t = awkward1.layout.PrimitiveType("float64", {"__class__": "Dummy"})
    x = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]), type=t)
    a = awkward1.Array(x, namespace=ns)
    assert repr(a) == "<Dummy [1.1, 2.2, 3.3, 4.4, 5.5]>"

    x2 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), x)
    a2 = awkward1.Array(x2, namespace=ns)
    assert repr(a2) == "<Array [[1.1, 2.2, 3.3], [], [4.4, 5.5]] type='3 * var * float64[parameters={\"__...'>"
    assert repr(a2[0]) == "<Dummy [1.1, 2.2, 3.3]>"
    assert repr(a2[1]) == "<Dummy []>"
    assert repr(a2[2]) == "<Dummy [4.4, 5.5]>"

numba = pytest.importorskip("numba")

def test_boxing():
    @numba.njit
    def f1(q):
        return 3.14

    @numba.njit
    def f2(q):
        return q

    t = awkward1.layout.UnknownType(parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.PrimitiveType("int32", parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.PrimitiveType("float64", parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.ListType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32"), parameters={"one": 1, "two": 2}))
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.ListType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.RegularType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")), 5, parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.OptionType(awkward1.layout.PrimitiveType("int32"), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.OptionType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.UnionType((awkward1.layout.PrimitiveType("int32"), awkward1.layout.PrimitiveType("float64")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.RecordType((awkward1.layout.PrimitiveType("int32"), awkward1.layout.PrimitiveType("float64")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.RecordType({"one": awkward1.layout.PrimitiveType("int32"), "two": awkward1.layout.PrimitiveType("float64")}, parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t
