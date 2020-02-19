# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")
numba_unsafe_refcount = pytest.importorskip("numba.unsafe.refcount")

def test():
    array = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))

    @numba.jit(nopython=True)
    def f1(x):
        return numba_unsafe_refcount.get_refcount(x.array)

    assert sys.getrefcount(array) == 2
    assert f1(array) == 1
    assert sys.getrefcount(array) == 2
