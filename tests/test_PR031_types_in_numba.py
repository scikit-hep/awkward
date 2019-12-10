# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools
import pickle

import pytest
import numpy

numba = pytest.importorskip("numba")

import awkward1

if sys.version_info[0] < 3:
    pytest.skip("pybind11 pickle, and hence numba serialization with types, only works in Python 3", allow_module_level=True)

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
    t = awkward1.utf8; assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.string; assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.ArrayType(awkward1.layout.PrimitiveType("int32"), 100); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.RegularType(awkward1.layout.PrimitiveType("int32"), 5); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.OptionType(awkward1.layout.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.UnionType(awkward1.layout.PrimitiveType("int32"), awkward1.layout.PrimitiveType("float64")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.layout.RecordType(one=awkward1.layout.PrimitiveType("int32"), two=awkward1.layout.PrimitiveType("float64")); assert pickle.loads(pickle.dumps(t)) == t

def test_boxing():
    @numba.njit
    def f1(q):
        return 3.14

    @numba.njit
    def f2(q):
        return q

    t = awkward1.layout.UnknownType()
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.PrimitiveType("int32")
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.PrimitiveType("float64")
    f1(t)
    assert f2(t) == t

    t = awkward1.utf8
    f1(t)
    assert f2(t) == t

    t = awkward1.string
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.PrimitiveType("float64")
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.ArrayType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")), 100)
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.ListType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")))
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.RegularType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")), 5)
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.OptionType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int32")))
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.UnionType(awkward1.layout.PrimitiveType("int32"), awkward1.layout.PrimitiveType("float64"))
    f1(t)
    assert f2(t) == t

    t = awkward1.layout.RecordType(one=awkward1.layout.PrimitiveType("int32"), two=awkward1.layout.PrimitiveType("float64"))
    f1(t)
    assert f2(t) == t

class D(awkward1.highlevel.Array):
    @staticmethod
    def typestr(baretype, parameters):
        return "D[{0}]".format(baretype)

def test_numpyarray():
    array1 = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5))
    dint64 = awkward1.layout.DressedType(awkward1.layout.PrimitiveType("int64"), D)
    array1.type = awkward1.layout.ArrayType(awkward1.layout.RegularType(awkward1.layout.RegularType(dint64, 5), 3), 2)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.baretype) == "2 * 3 * 5 * int64"
    assert repr(array2.type) == "2 * 3 * 5 * D[int64]"
    assert repr(array2[0].type) == "3 * 5 * D[int64]"
    assert repr(array2[0, 0].type) == "5 * D[int64]"
    assert array2[-1, -1, -1] == 29

def test_regulararray():
    array1 = awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.arange(10, dtype=numpy.int64)), 5)
    dregint64 = awkward1.layout.DressedType(awkward1.layout.RegularType(awkward1.layout.PrimitiveType("int64"), 5), D)
    array1.type = awkward1.layout.ArrayType(dregint64, 2)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.baretype) == "2 * 5 * int64"
    assert repr(array2.type) == "2 * D[5 * int64]"

def test_listoffsetarray():
    array1 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)))
    dvarint64 = awkward1.layout.DressedType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int64")), D)
    array1.type = awkward1.layout.ArrayType(dvarint64, 3)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.baretype) == "3 * var * int64"
    assert repr(array2.type) == "3 * D[var * int64]"

def test_listarray():
    array1 = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3], dtype=numpy.int64)), awkward1.layout.Index64(numpy.array([3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64)))
    dvarint64 = awkward1.layout.DressedType(awkward1.layout.ListType(awkward1.layout.PrimitiveType("int64")), D)
    array1.type = awkward1.layout.ArrayType(dvarint64, 3)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.baretype) == "3 * var * int64"
    assert repr(array2.type) == "3 * D[var * int64]"

def test_recordarray():
    array1 = awkward1.Array([{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}]).layout
    dvarrec = awkward1.layout.DressedType(array1.type.nolength(), D)
    array1.type = awkward1.layout.ArrayType(dvarrec, 3)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.baretype) in ("3 * {'one': int64, 'two': float64}", "3 * {'two': float64, 'one': int64}")
    assert repr(array2.type) in ("3 * D[{'one': int64, 'two': float64}]", "3 * D[{'two': float64, 'one': int64}]")

    assert repr(array2[0].baretype) in ("{'one': int64, 'two': float64}", "{'two': float64, 'one': int64}")
    assert repr(array2[0].type) in ("D[{'one': int64, 'two': float64}]", "D[{'two': float64, 'one': int64}]")
