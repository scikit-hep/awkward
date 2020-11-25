# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


numba = pytest.importorskip("numba")


def test():
    @numba.njit
    def do_something(array):
        out = numpy.zeros(len(array), numpy.bool_)
        for i, x in enumerate(array):
            if x:
                out[i] = x
        return out

    array = awkward1.Array([True, False, False])
    assert do_something(array).tolist() == [True, False, False]
