# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools
import pickle

import pytest
import numpy

numba = pytest.importorskip("numba")

import awkward1

def test_from_lookup():
    r = awkward1.layout.RecordArray.from_lookup([awkward1.layout.EmptyArray(), awkward1.layout.EmptyArray()], {"one": 0, "two": 1})
    assert r.lookup == {"one": 0, "two": 1}
    assert r.reverselookup == ["one", "two"]

    r = awkward1.layout.RecordArray.from_lookup([awkward1.layout.EmptyArray(), awkward1.layout.EmptyArray()], {"one": 0, "two": 1}, ["uno", "dos"])
    assert r.lookup == {"one": 0, "two": 1}
    assert r.reverselookup == ["uno", "dos"]

    r = awkward1.layout.RecordType.from_lookup([awkward1.layout.UnknownType(), awkward1.layout.UnknownType()], {"one": 0, "two": 1})
    assert r.lookup == {"one": 0, "two": 1}
    assert r.reverselookup == ["one", "two"]

    r = awkward1.layout.RecordType.from_lookup([awkward1.layout.UnknownType(), awkward1.layout.UnknownType()], {"one": 0, "two": 1}, ["uno", "dos"])
    assert r.lookup == {"one": 0, "two": 1}
    assert r.reverselookup == ["uno", "dos"]

def test_pickle():
    t = awkward1.layout.UnknownType(); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.PrimitiveType("int32"); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.PrimitiveType("float64"); assert pickle.loads(pickle.dumps(t)) == t
    # t = awkward1.layout.DressedType; assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.ArrayType(awkward1.layout.PrimitiveType("int32"), 100); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.RegularType(awkward1.layout.PrimitiveType("int32"), 5); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.OptionType(awkward1.layout.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.UnionType(awkward1.layout.PrimitiveType("int32"), awkward1.layout.PrimitiveType("float64")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.RecordType(one=awkward1.layout.PrimitiveType("int32"), two=awkward1.layout.PrimitiveType("float64")); assert pickle.loads(pickle.dumps(t)) == t

# class DressedType(TypeType):


# def test_boxing():
#     t = awkward1.layout.ArrayType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64")), 10)
#
#     @numba.njit
#     def f1(q):
#         return 3.14
#
#     f1(t)
#
#     @numba.njit
#     def f2(q):
#         return q
#
#     assert f2(t) == t
