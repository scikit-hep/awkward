# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

ak._v2.numba.register_and_check()


def test():
    @numba.njit
    def f1(array):
        return array.ndim

    assert f1(ak._v2.highlevel.Array([[1, 2, 3], [], [4, 5]])) == 2
    assert f1(ak._v2.highlevel.Array([[[1], [2, 3]], [], [[4, 5], []]])) == 3

    with pytest.raises(numba.core.errors.TypingError):
        f1(ak._v2.highlevel.Record({"x": [1, 2, 3], "y": [4]}))
