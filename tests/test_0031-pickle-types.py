# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import sys
import pickle

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


@pytest.mark.skipif(ak._util.py27, reason="pybind11 pickle only works in Python 3")
def test_pickle():
    t = ak.types.UnknownType()
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.PrimitiveType("int32")
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.PrimitiveType("float64")
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.ArrayType(ak.types.PrimitiveType("int32"), 100)
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.ListType(ak.types.PrimitiveType("int32"))
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.RegularType(ak.types.PrimitiveType("int32"), 5)
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.OptionType(ak.types.PrimitiveType("int32"))
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.UnionType(
        (ak.types.PrimitiveType("int32"), ak.types.PrimitiveType("float64"))
    )
    assert pickle.loads(pickle.dumps(t)) == t
    t = ak.types.RecordType(
        {
            "one": ak.types.PrimitiveType("int32"),
            "two": ak.types.PrimitiveType("float64"),
        }
    )
    assert pickle.loads(pickle.dumps(t)) == t
