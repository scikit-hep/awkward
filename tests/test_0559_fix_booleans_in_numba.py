# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test():
    @numba.njit
    def do_something(array):
        out = np.zeros(len(array), np.bool_)
        for i, x in enumerate(array):
            if x:
                out[i] = x
        return out

    array = ak.highlevel.Array([True, False, False])
    assert do_something(array).tolist() == [True, False, False]
