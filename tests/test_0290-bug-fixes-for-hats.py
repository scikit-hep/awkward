# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

numba = pytest.importorskip("numba")

def test_unmasked():
    @numba.njit
    def find_it(array):
        for item in array:
            if item is None:
                pass
            elif item.x == 3:
                return item
        return None

    content = awkward1.Array([{"x": 1}, {"x": 2}, {"x": 3}]).layout
    unmasked = awkward1.layout.UnmaskedArray(content)
    array = awkward1.Array(unmasked)
    assert awkward1.to_list(find_it(array)) == {"x": 3}

def test_indexedoption():
    @numba.njit
    def find_it(array):
        for item in array:
            if item is None:
                pass
            elif item.x == 3:
                return item
        return None

    array = awkward1.Array([{"x": 1}, {"x": 2}, None, {"x": 3}])
    assert awkward1.to_list(find_it(array)) == {"x": 3}

def test_indexed_1():
    @numba.njit
    def f1(array, check):
        for i in range(len(array)):
            item = array[i]
            if item.x == check:
                return i
        return 999

    content = awkward1.Array([{"x": 100}, {"x": 101}, {"x": 102}]).layout
    index = awkward1.layout.Index64(numpy.array([2, 0, 1], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    array = awkward1.Array(indexedarray)

    assert f1(array, 100) == 1
    assert f1(array, 101) == 2
    assert f1(array, 102) == 0
    assert f1(array, 12345) == 999

def test_indexed_2():
    @numba.njit
    def f1(array, check):
        for item in array:
            if item.x == check:
                return item
        return None

    content = awkward1.Array([{"x": 100}, {"x": 101}, {"x": 102}]).layout
    index = awkward1.layout.Index64(numpy.array([2, 0, 1], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    array = awkward1.Array(indexedarray)

    assert f1(array, 100) == {"x": 100}
    assert f1(array, 101) == {"x": 101}
    assert f1(array, 102) == {"x": 102}
    assert f1(array, 12345) == None
