# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools
import pickle

import pytest
import numpy

import awkward1

if sys.version_info[0] < 3:
    pytest.skip("pybind11 pickle only works in Python 3", allow_module_level=True)

def test_pickle():
    t = awkward1.types.UnknownType(); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.PrimitiveType("int32"); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.PrimitiveType("float64"); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.ArrayType(awkward1.types.PrimitiveType("int32"), 100); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.ListType(awkward1.types.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.RegularType(awkward1.types.PrimitiveType("int32"), 5); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.OptionType(awkward1.types.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.UnionType((awkward1.types.PrimitiveType("int32"), awkward1.types.PrimitiveType("float64"))); assert pickle.loads(pickle.dumps(t)) == t
    t = awkward1.types.RecordType({"one": awkward1.types.PrimitiveType("int32"), "two": awkward1.types.PrimitiveType("float64")}); assert pickle.loads(pickle.dumps(t)) == t
