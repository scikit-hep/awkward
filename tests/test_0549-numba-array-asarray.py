# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


numba = pytest.importorskip("numba")


def test_asarray():
    @numba.njit
    def f1(x):
        return numpy.array(x)

    assert isinstance(f1(awkward1.Array([[[1], [2], [3]], [[4], [5], [6]]])), numpy.ndarray)
    assert f1(awkward1.Array([[[1], [2], [3]], [[4], [5], [6]]])).tolist() == [[[1], [2], [3]], [[4], [5], [6]]]
    assert f1(awkward1.Array([[1, 2, 3], [4, 5, 6]])).tolist() == [[1, 2, 3], [4, 5, 6]]
    assert f1(awkward1.Array([1, 2, 3, 4, 5, 6])).tolist() == [1, 2, 3, 4, 5, 6]

    with pytest.raises(ValueError):
        f1(awkward1.Array([[1, 2, 3, 4], [5, 6]]))
