# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools
import pickle

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

def test_types_with_parameters():
    t = awkward1.types.UnknownType()
    assert t.parameters == {}
    t.parameters = {"key": ["val", "ue"]}
    assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.UnknownType(parameters={"key": ["val", "ue"]})
    assert t.parameters == {"key": ["val", "ue"]}

    t = awkward1.types.PrimitiveType("int32", parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.PrimitiveType("float64", parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.ArrayType(awkward1.types.PrimitiveType("int32"), 100, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.ListType(awkward1.types.PrimitiveType("int32"), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.RegularType(awkward1.types.PrimitiveType("int32"), 5, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.OptionType(awkward1.types.PrimitiveType("int32"), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.UnionType((awkward1.types.PrimitiveType("int32"), awkward1.types.PrimitiveType("float64")), parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}
    t = awkward1.types.RecordType({"one": awkward1.types.PrimitiveType("int32"), "two": awkward1.types.PrimitiveType("float64")}, parameters={"key": ["val", "ue"]}); assert t.parameters == {"key": ["val", "ue"]}

    t = awkward1.types.UnknownType(parameters={"key1": ["val", "ue"], "key2": u"one \u2192 two"})
    assert t.parameters == {"key2": u"one \u2192 two", "key1": ["val", "ue"]}

    assert t == awkward1.types.UnknownType(parameters={"key2": u"one \u2192 two", "key1": ["val", "ue"]})
    assert t != awkward1.types.UnknownType(parameters={"key": ["val", "ue"]})

def test_dress():
    class Dummy(awkward1.highlevel.Array):
        def __repr__(self):
            return "<Dummy {0}>".format(str(self))
    ns = {"Dummy": Dummy}

    x = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    x.setparameter("__array__", "Dummy")
    x.setparameter("__typestr__", "D[5 * float64]")
    a = awkward1.Array(x, behavior=ns)
    assert repr(a) == "<Dummy [1.1, 2.2, 3.3, 4.4, 5.5]>"

    x2 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"__array__": "Dummy"}))
    a2 = awkward1.Array(x2, behavior=ns)
    assert repr(a2) == "<Array [<Dummy [1.1, 2.2, 3.3]>, ... ] type='3 * var * float64[parameters={\"__ar...'>"
    assert repr(a2[0]) == "<Dummy [1.1, 2.2, 3.3]>"
    assert repr(a2[1]) == "<Dummy []>"
    assert repr(a2[2]) == "<Dummy [4.4, 5.5]>"

def test_typestr():
    t = awkward1.types.PrimitiveType("float64", {"__typestr__": "something"})
    t2 = awkward1.types.ListType(t)

    assert repr(t) == "something"
    assert repr(t2) == "var * something"

def test_record_name():
    builder = awkward1.layout.ArrayBuilder()

    builder.beginrecord("Dummy")
    builder.field("one")
    builder.integer(1)
    builder.field("two")
    builder.real(1.1)
    builder.endrecord()

    builder.beginrecord("Dummy")
    builder.field("two")
    builder.real(2.2)
    builder.field("one")
    builder.integer(2)
    builder.endrecord()

    a = builder.snapshot()
    assert repr(a.type) == 'struct[["one", "two"], [int64, float64], parameters={"__record__": "Dummy"}]'
    assert a.type.parameters == {"__record__": "Dummy"}

def test_builder_string():
    builder = awkward1.ArrayBuilder()

    builder.bytestring(b"one")
    builder.bytestring(b"two")
    builder.bytestring(b"three")

    a = builder.snapshot()
    if py27:
        assert str(a) == "['one', 'two', 'three']"
    else:
        assert str(a) == "[b'one', b'two', b'three']"
    assert awkward1.tolist(a) == [b'one', b'two', b'three']
    assert awkward1.tojson(a) == '["one","two","three"]'
    if py27:
        assert repr(a) == "<Array ['one', 'two', 'three'] type='3 * bytes'>"
    else:
        assert repr(a) == "<Array [b'one', b'two', b'three'] type='3 * bytes'>"
    assert repr(a.type) == "3 * bytes"

    builder = awkward1.ArrayBuilder()

    builder.string("one")
    builder.string("two")
    builder.string("three")

    a = builder.snapshot()
    if py27:
        assert str(a) == "[u'one', u'two', u'three']"
    else:
        assert str(a) == "['one', 'two', 'three']"
    assert awkward1.tolist(a) == ['one', 'two', 'three']
    assert awkward1.tojson(a) == '["one","two","three"]'
    if py27:
        assert repr(a) == "<Array [u'one', u'two', u'three'] type='3 * string'>"
    else:
        assert repr(a) == "<Array ['one', 'two', 'three'] type='3 * string'>"
    assert repr(a.type) == "3 * string"

    builder = awkward1.ArrayBuilder()

    builder.beginlist()
    builder.string("one")
    builder.string("two")
    builder.string("three")
    builder.endlist()

    builder.beginlist()
    builder.endlist()

    builder.beginlist()
    builder.string("four")
    builder.string("five")
    builder.endlist()

    a = builder.snapshot()
    if py27:
        assert str(a) == "[[u'one', u'two', u'three'], [], [u'four', u'five']]"
    else:
        assert str(a) == "[['one', 'two', 'three'], [], ['four', 'five']]"
    assert awkward1.tolist(a) == [['one', 'two', 'three'], [], ['four', 'five']]
    assert awkward1.tojson(a) == '[["one","two","three"],[],["four","five"]]'
    assert repr(a.type) == "3 * var * string"

def test_fromiter_fromjson():
    assert awkward1.tolist(awkward1.fromiter(["one", "two", "three"])) == ["one", "two", "three"]
    assert awkward1.tolist(awkward1.fromiter([["one", "two", "three"], [], ["four", "five"]])) == [["one", "two", "three"], [], ["four", "five"]]

    assert awkward1.tolist(awkward1.fromjson('["one", "two", "three"]')) == ["one", "two", "three"]
    assert awkward1.tolist(awkward1.fromjson('[["one", "two", "three"], [], ["four", "five"]]')) == [["one", "two", "three"], [], ["four", "five"]]
