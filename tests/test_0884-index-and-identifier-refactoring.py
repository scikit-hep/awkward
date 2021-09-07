# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np

import awkward as ak

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_index32():
    cpp_array = ak.layout.Index32(np.zeros(10))
    py_array = ak._v2.index.Index.zeros(10, np, np.int32)

    assert len(cpp_array) == len(py_array)
    assert "i32" == py_array.form


def test_index64():
    cpp_array = ak.layout.Index64(np.zeros(10))
    py_array = ak._v2.index.Index.zeros(10, np, np.int64)

    assert len(cpp_array) == len(py_array)
    assert "i64" == py_array.form


def test_identifier32():
    cpp_array = ak.layout.Identities32(123, [(1, "one"), (2, "two")], 10, 5)
    py_array = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int32
    )

    assert len(cpp_array) == len(py_array)
    assert py_array._data.dtype == np.dtype(np.int32)


def test_identifier64():
    cpp_array = ak.layout.Identities64(123, [(1, "one"), (2, "two")], 10, 5)
    py_array = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )

    assert len(cpp_array) == len(py_array)
    assert py_array._data.dtype == np.dtype(np.int64)
