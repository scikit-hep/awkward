# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test():
    @numba.njit
    def f1(array):
        return array.ndim

    assert f1(ak.highlevel.Array([[1, 2, 3], [], [4, 5]])) == 2
    assert f1(ak.highlevel.Array([[[1], [2, 3]], [], [[4, 5], []]])) == 3

    with pytest.raises(numba.core.errors.TypingError):
        f1(ak.highlevel.Record({"x": [1, 2, 3], "y": [4]}))
