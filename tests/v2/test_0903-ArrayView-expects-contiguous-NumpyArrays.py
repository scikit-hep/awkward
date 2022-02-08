# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


numba = pytest.importorskip("numba")

ak._v2.numba.register_and_check()


def test():
    @numba.njit
    def f1(x):
        return x[0], x[1]

    array = ak._v2.highlevel.Array(np.arange(4).reshape(2, 2)[:, 0])

    assert f1.py_func(array) == (0, 2)
    assert f1(array) == (0, 2)
