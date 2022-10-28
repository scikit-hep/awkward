# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test_index32():
    py_array = ak.index.Index.zeros(10, ak.nplikes.Numpy.instance(), np.int32)

    assert len(py_array) == 10
    assert "i32" == py_array.form


def test_index64():
    py_array = ak.index.Index.zeros(10, ak.nplikes.Numpy.instance(), np.int64)

    assert len(py_array) == 10
    assert "i64" == py_array.form
