# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import re
import awkward as ak
import awkward._v2.index as ain
import awkward._v2.identifier as aid


def test_index32():
    cpp_array = ak.layout.Index32(np.zeros(10))
    py_array = ain.Index.zeros(10, np, np.int32)

    assert cpp_array.__len__() == py_array.__len__()
    assert "i32" == py_array.form()

    assert py_array.__repr__() == "[0 0 0 0 0 0 0 0 0 0]"

    assert cpp_array.__repr__().split('"')[1] == "[0 0 0 0 0 0 0 0 0 0]"


def test_index64():
    cpp_array = ak.layout.Index64(np.zeros(10))
    py_array = ain.Index.zeros(10, np, np.int64)

    assert cpp_array.__len__() == py_array.__len__()
    assert "i64" == py_array.form()

    assert py_array.__repr__() == "[0 0 0 0 0 0 0 0 0 0]"

    assert cpp_array.__repr__().split('"')[1] == "[0 0 0 0 0 0 0 0 0 0]"


def test_identifier32():
    cpp_array = ak.layout.Identities32(123, [(1, "one"), (2, "two")], 10, 5)
    py_array = aid.Identifier.zeros(123, [(1, "one"), (2, "two")], 5, 10, np, np.int32)

    assert cpp_array.__len__() == py_array.__len__()
    assert py_array._data.dtype == np.dtype(np.int32)

    assert (
        py_array._repr("", "", "").split(" at")[0]
        == re.sub(r" offset=\"\d*\"", "", cpp_array.__repr__()).split(" at")[0]
    )


def test_identifier64():
    cpp_array = ak.layout.Identities64(123, [(1, "one"), (2, "two")], 10, 5)
    py_array = aid.Identifier.zeros(123, [(1, "one"), (2, "two")], 5, 10, np, np.int64)

    assert cpp_array.__len__() == py_array.__len__()
    assert py_array._data.dtype == np.dtype(np.int64)

    assert (
        py_array._repr("", "", "").split(" at")[0]
        == re.sub(r" offset=\"\d*\"", "", cpp_array.__repr__()).split(" at")[0]
    )
