# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools
import pickle

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

pytest.skip("Disabling Numba until the rewrite is done.", allow_module_level=True)

numba = pytest.importorskip("numba")

def test_record_name_numba():
    @numba.njit
    def f1(fillable):
        fillable.beginrecord("Dummy")
        fillable.field("one")
        fillable.integer(1)
        fillable.field("two")
        fillable.real(1.1)
        fillable.endrecord()

        fillable.beginrecord("Dummy")
        fillable.field("two")
        fillable.real(2.2)
        fillable.field("one")
        fillable.integer(2)
        fillable.endrecord()

    fillable = awkward1.layout.FillableArray()
    f1(fillable)

    a = fillable.snapshot()
    assert repr(a.type) == 'struct[["one", "two"], [int64, float64], parameters={"__record__": "Dummy"}]'
    assert a.type.parameters == {"__record__": "Dummy"}

def test_boxing():
    @numba.njit
    def f1(q):
        return 3.14

    @numba.njit
    def f2(q):
        return q

    t = awkward1.types.UnknownType(parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.PrimitiveType("int32", parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.PrimitiveType("float64", parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.ListType(awkward1.types.ListType(awkward1.types.PrimitiveType("int32"), parameters={"one": 1, "two": 2}))
    f1(t)
    assert f2(t) == t

    t = awkward1.types.ListType(awkward1.types.ListType(awkward1.types.PrimitiveType("int32")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.RegularType(awkward1.types.ListType(awkward1.types.PrimitiveType("int32")), 5, parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.OptionType(awkward1.types.PrimitiveType("int32"), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.OptionType(awkward1.types.ListType(awkward1.types.PrimitiveType("int32")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.UnionType((awkward1.types.PrimitiveType("int32"), awkward1.types.PrimitiveType("float64")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.RecordType((awkward1.types.PrimitiveType("int32"), awkward1.types.PrimitiveType("float64")), parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t

    t = awkward1.types.RecordType({"one": awkward1.types.PrimitiveType("int32"), "two": awkward1.types.PrimitiveType("float64")}, parameters={"one": 1, "two": 2})
    f1(t)
    assert f2(t) == t
