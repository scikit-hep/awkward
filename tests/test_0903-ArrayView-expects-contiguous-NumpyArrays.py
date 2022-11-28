# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test():
    @numba.njit
    def f1(x):
        return x[0], x[1]

    array = ak.highlevel.Array(np.arange(4).reshape(2, 2)[:, 0])

    assert f1.py_func(array) == (0, 2)
    assert f1(array) == (0, 2)
