# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


numba = pytest.importorskip("numba")


def test():
    @numba.njit
    def do_something(array):
        out = np.zeros(len(array), np.bool_)
        for i, x in enumerate(array):
            if x:
                out[i] = x
        return out

    array = ak.Array([True, False, False])
    assert do_something(array).tolist() == [True, False, False]
