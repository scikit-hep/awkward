# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")
numba_unsafe_refcount = pytest.importorskip("numba.unsafe.refcount")

def test_NumpyArray():
    array = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    # assert sys.getrefcount(array) == 2

    # @numba.njit
    # def f1(x):
    #     return numba_unsafe_refcount.get_refcount(x.array)

    # assert f1(array) == 1
    # assert sys.getrefcount(array) == 2

    # @numba.njit
    # def f2(x):
    #     return numba_unsafe_refcount.get_refcount(x.array), x

    # assert f2(array)[0] == 1
    # assert sys.getrefcount(array) == 2

    # @numba.njit
    # def f3(x):
    #     return numba_unsafe_refcount.get_refcount(x.array), len(x)

    # assert f3(array) == (1, 5)
    # assert sys.getrefcount(array) == 2

    # @numba.njit
    # def f4(x):
    #     return numba_unsafe_refcount.get_refcount(x.array), x[1]

    # assert f4(array) == (1, 2.2)
    # assert sys.getrefcount(array) == 2

    @numba.njit
    def f5(x):
        return x[1:4]

    # assert awkward1.tolist(f5(array)) == [2.2, 3.3, 4.4]
    print(f5(array))

    raise Exception

    # array2 = awkward1.layout.NumpyArray(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    # assert awkward1.tolist(f4(array2)[1]) == [4.4, 5.5, 6.6]
