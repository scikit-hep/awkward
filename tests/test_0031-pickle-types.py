# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import pickle

import pytest
import numpy as np
import awkward1 as ak

if sys.version_info[0] < 3:
    pytest.skip("pybind11 pickle only works in Python 3", allow_module_level=True)

def test_pickle():
    t = ak.types.UnknownType(); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.PrimitiveType("int32"); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.PrimitiveType("float64"); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.ArrayType(ak.types.PrimitiveType("int32"), 100); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.ListType(ak.types.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.RegularType(ak.types.PrimitiveType("int32"), 5); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.OptionType(ak.types.PrimitiveType("int32")); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.UnionType((ak.types.PrimitiveType("int32"), ak.types.PrimitiveType("float64"))); assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.RecordType({"one": ak.types.PrimitiveType("int32"), "two": ak.types.PrimitiveType("float64")}); assert pickle.loads(pickle.dumps(t)) == t
