# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

def test_ByteMaskedArray():
    content = awkward1.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]).layout
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 0], dtype=numpy.int8))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, content, validwhen=False))
    assert awkward1.tolist(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, awkward1.layout.ByteMaskedArray)
    assert awkward1.tolist(y) == awkward1.tolist(array)

    @numba.njit
    def f3(x, i):
        return x[i]

    assert awkward1.tolist(f3(array, 0)) == [0.0, 1.1, 2.2]
    assert awkward1.tolist(f3(array, 1)) == []
    assert f3(array, 2) is None
    assert f3(array, 3) is None
    assert awkward1.tolist(f3(array, 4)) == [6.6, 7.7, 8.8, 9.9]

def test_BitMaskedArray():
    content = awkward1.layout.NumpyArray(numpy.arange(13))
    mask = awkward1.layout.IndexU8(numpy.array([58, 59], dtype=numpy.uint8))
    array = awkward1.Array(awkward1.layout.BitMaskedArray(mask, content, validwhen=True, length=13, lsb_order=True))
    assert numpy.asarray(array.layout.bytemask()).tolist() == [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    assert numpy.asarray(array.layout.toByteMaskedArray().mask).tolist() == [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    assert numpy.asarray(array.layout.toIndexedOptionArray64().index).tolist() == [-1, 1, -1, 3, 4, 5, -1, -1, 8, 9, -1, 11, 12]
    assert awkward1.tolist(array) == [None, 1, None, 3, 4, 5, None, None, 8, 9, None, 11, 12]

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, awkward1.layout.BitMaskedArray)
    assert awkward1.tolist(y) == awkward1.tolist(array)

def test_UnmaskedArray():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.float64))
    array = awkward1.Array(awkward1.layout.UnmaskedArray(content))
    assert awkward1.tolist(array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert str(awkward1.typeof(array)) == "5 * ?float64"

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, awkward1.layout.UnmaskedArray)
    assert awkward1.tolist(y) == awkward1.tolist(array)
    assert str(awkward1.typeof(y)) == str(awkward1.typeof(array))
