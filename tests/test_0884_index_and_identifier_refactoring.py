# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.numpy import Numpy

numpy = Numpy.instance()


def test_index32():
    py_array = ak.index.Index.zeros(10, nplike=numpy.instance(), dtype=np.int32)

    assert len(py_array) == 10
    assert "i32" == py_array.form


def test_index64():
    py_array = ak.index.Index.zeros(10, nplike=numpy.instance(), dtype=np.int64)

    assert len(py_array) == 10
    assert "i64" == py_array.form
