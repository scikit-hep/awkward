# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

def test_ByteMaskedArray():
    content = awkward1.from_iter([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False)
    mask = awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 0], dtype=numpy.int8))
    array = awkward1.Array(awkward1.layout.ByteMaskedArray(mask, content, valid_when=False))
    assert awkward1.to_list(array) == [[0.0, 1.1, 2.2], [], None, None, [6.6, 7.7, 8.8, 9.9]]

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, awkward1.layout.ByteMaskedArray)
    assert awkward1.to_list(y) == awkward1.to_list(array)

    @numba.njit
    def f3(x, i):
        return x[i]

    assert awkward1.to_list(f3(array, 0)) == [0.0, 1.1, 2.2]
    assert awkward1.to_list(f3(array, 1)) == []
    assert f3(array, 2) is None
    assert f3(array, 3) is None
    assert awkward1.to_list(f3(array, 4)) == [6.6, 7.7, 8.8, 9.9]

def test_BitMaskedArray():
    content = awkward1.layout.NumpyArray(numpy.arange(13))
    mask = awkward1.layout.IndexU8(numpy.array([58, 59], dtype=numpy.uint8))
    array = awkward1.Array(awkward1.layout.BitMaskedArray(mask, content, valid_when=True, length=13, lsb_order=True))
    assert awkward1.to_list(array) == [None, 1, None, 3, 4, 5, None, None, 8, 9, None, 11, 12]

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, awkward1.layout.BitMaskedArray)
    assert awkward1.to_list(y) == awkward1.to_list(array)

    @numba.njit
    def f3(x, i):
        return x[i]

    assert [f3(array, i) for i in range(len(array))] == [None, 1, None, 3, 4, 5, None, None, 8, 9, None, 11, 12]

    array = awkward1.Array(awkward1.layout.BitMaskedArray(mask, content, valid_when=True, length=13, lsb_order=False))
    assert awkward1.to_list(array) == [None, None, 2, 3, 4, None, 6, None, None, None, 10, 11, 12]

    assert [f3(array, i) for i in range(len(array))] == [None, None, 2, 3, 4, None, 6, None, None, None, 10, 11, 12]

def test_UnmaskedArray():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.float64))
    array = awkward1.Array(awkward1.layout.UnmaskedArray(content))
    assert awkward1.to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert str(awkward1.type(array)) == "5 * ?float64"

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, awkward1.layout.UnmaskedArray)
    assert awkward1.to_list(y) == awkward1.to_list(array)
    assert str(awkward1.type(y)) == str(awkward1.type(array))

    @numba.njit
    def f3(x, i):
        return x[i]

    assert [f3(array, i) for i in range(len(array))] == [1.1, 2.2, 3.3, 4.4, 5.5]
