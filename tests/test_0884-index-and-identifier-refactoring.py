# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_index32():
    py_array = ak.index.Index.zeros(
        10, nplike=ak.nplikes.Numpy.instance(), dtype=np.int32, index_is_fixed=True
    )

    assert len(py_array) == 10
    assert "i32" == py_array.form


def test_index64():
    py_array = ak.index.Index.zeros(
        10, nplike=ak.nplikes.Numpy.instance(), dtype=np.int64, index_is_fixed=True
    )

    assert len(py_array) == 10
    assert "i64" == py_array.form
