# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools
import pickle

import pytest
import numpy

pytest.skip("Disabling Numba until the rewrite is done.", allow_module_level=True)

numba = pytest.importorskip("numba")

import awkward1

if sys.version_info[0] < 3:
    pytest.skip("pybind11 pickle, and hence numba serialization with types, only works in Python 3", allow_module_level=True)

def test_pickle():
    t = awkward1.types.UnknownType(); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.PrimitiveType("int32"); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.PrimitiveType("float64"); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.utf8; assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.string; assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.ArrayType(awkward1.types.PrimitiveType("int32"), 100); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.ListType(awkward1.types.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.RegularType(awkward1.types.PrimitiveType("int32"), 5); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.OptionType(awkward1.types.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.UnionType((awkward1.types.PrimitiveType("int32"), awkward1.types.PrimitiveType("float64"))); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.RecordType({"one": awkward1.types.PrimitiveType("int32"), "two": awkward1.types.PrimitiveType("float64")}); assert pickle.loads(pickle.dumps(t)) == t

def test_boxing():
    @numba.njit
    def f1(q):
        return 3.14

    @numba.njit
    def f2(q):
        return q

    t = awkward1.types.UnknownType()
    f1(t)
    assert f2(t) == t

    t = awkward1.types.PrimitiveType("int32")
    f1(t)
    assert f2(t) == t

    t = awkward1.types.PrimitiveType("float64")
    f1(t)
    assert f2(t) == t

    t = awkward1.utf8
    f1(t)
    assert f2(t) == t

    t = awkward1.string
    f1(t)
    assert f2(t) == t

    t = awkward1.types.PrimitiveType("float64")
    f1(t)
    assert f2(t) == t

    t = awkward1.types.ListType(awkward1.types.ListType(awkward1.types.PrimitiveType("int32")))
    f1(t)
    assert f2(t) == t

    t = awkward1.types.RegularType(awkward1.types.ListType(awkward1.types.PrimitiveType("int32")), 5)
    f1(t)
    assert f2(t) == t

    t = awkward1.types.OptionType(awkward1.types.ListType(awkward1.types.PrimitiveType("int32")))
    f1(t)
    assert f2(t) == t

    t = awkward1.types.UnionType((awkward1.types.PrimitiveType("int32"), awkward1.types.PrimitiveType("float64")))
    f1(t)
    assert f2(t) == t

    t = awkward1.types.RecordType({"one": awkward1.types.PrimitiveType("int32"), "two": awkward1.types.PrimitiveType("float64")})
    f1(t)
    assert f2(t) == t

class D(awkward1.highlevel.Array):
    pass

def test_numpyarray():
    dint64 = awkward1.types.PrimitiveType("int64", {"__record__": "D", "__typestr__": "D[int64]"})
    array1 = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5)).astype(awkward1.types.RegularType(awkward1.types.RegularType(dint64, 5), 3))

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "3 * 5 * D[int64]"
    assert repr(array2[0].type) == "5 * D[int64]"
    assert repr(array2[0, 0].type) == "D[int64]"
    assert array2[-1, -1, -1] == 29

def test_regulararray():
    dregint64 = awkward1.types.RegularType(awkward1.types.PrimitiveType("int64"), 5, {"__record__": "D", "__typestr__": "D[5 * int64]"})
    array1 = awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.arange(10, dtype=numpy.int64)), 5).astype(dregint64)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[5 * int64]"

def test_listoffsetarray():
    dvarint64 = awkward1.types.ListType(awkward1.types.PrimitiveType("int64"), {"__record__": "D", "__typestr__": "D[var * int64]"})
    array1 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))).astype(dvarint64)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[var * int64]"

def test_listarray():
    dvarint64 = awkward1.types.ListType(awkward1.types.PrimitiveType("int64"), {"__record__": "D", "__typestr__": "D[var * int64]"})
    array1 = awkward1.layout.ListArray64(awkward1.layout.Index64(numpy.array([0, 3, 3], dtype=numpy.int64)), awkward1.layout.Index64(numpy.array([3, 3, 5], dtype=numpy.int64)), awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))).astype(dvarint64)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) == "D[var * int64]"

def test_recordarray():
    dvarrec = awkward1.types.RecordType({"one": awkward1.types.PrimitiveType("int64"), "two": awkward1.types.PrimitiveType("float64")}, {"__record__": "D", "__typestr__": "D[{\"one\": int64, \"two\": float64}]"})
    array1 = awkward1.Array([{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}]).layout.astype(dvarrec)

    @numba.njit
    def f1(q):
        return q

    array2 = f1(array1)

    assert repr(array2.type) in ('D[{"one": int64, "two": float64}]', ' D[{"two": float64, "one": int64}]')
    assert repr(array2[0].type) in ('D[{"one": int64, "two": float64}]', 'D[{"two": float64, "one": int64}]')
