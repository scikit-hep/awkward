# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_index32():
    py_array = ak._v2.index.Index.zeros(10, ak.nplike.Numpy.instance(), np.int32)

    assert len(py_array) == 10
    assert "i32" == py_array.form


def test_index64():
    py_array = ak._v2.index.Index.zeros(10, ak.nplike.Numpy.instance(), np.int64)

    assert len(py_array) == 10
    assert "i64" == py_array.form


def test_identifier32():
    py_array = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int32
    )

    assert len(py_array) == 5
    assert py_array._data.dtype == np.dtype(np.int32)


def test_identifier64():
    py_array = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )

    assert len(py_array) == 5
    assert py_array._data.dtype == np.dtype(np.int64)
