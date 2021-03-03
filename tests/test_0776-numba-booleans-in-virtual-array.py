# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


numba = pytest.importorskip("numba")


def test():
    @numba.njit
    def f1(array):
        return array[2][0].x

    array = ak.virtual(
        lambda: ak.Array([[{"x": True}, {"x": False}], [], [{"x": True}]])
    )
    assert f1(array) is True
