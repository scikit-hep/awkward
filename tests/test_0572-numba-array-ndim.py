# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")


def test():
    @numba.njit
    def f1(array):
        return array.ndim

    assert f1(ak.Array([[1, 2, 3], [], [4, 5]])) == 2
    assert f1(ak.Array([[[1], [2, 3]], [], [[4, 5], []]])) == 3

    with pytest.raises(numba.core.errors.TypingError):
        f1(ak.Record({"x": [1, 2, 3], "y": [4]}))


    partitioned = ak.partitioned([ak.Array([[1, 2, 3], [], [4, 5]]) for i in range(5)])

    assert f1(partitioned) == 2
