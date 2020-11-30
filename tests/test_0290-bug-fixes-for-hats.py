# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401

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

    content = ak.Array([{"x": 1}, {"x": 2}, {"x": 3}]).layout
    unmasked = ak.layout.UnmaskedArray(content)
    array = ak.Array(unmasked)
    assert ak.to_list(find_it(array)) == {"x": 3}


def test_indexedoption():
    @numba.njit
    def find_it(array):
        for item in array:
            if item is None:
                pass
            elif item.x == 3:
                return item
        return None

    array = ak.Array([{"x": 1}, {"x": 2}, None, {"x": 3}])
    assert ak.to_list(find_it(array)) == {"x": 3}


def test_indexed_1():
    @numba.njit
    def f1(array, check):
        for i in range(len(array)):
            item = array[i]
            if item.x == check:
                return i
        return 999

    content = ak.Array([{"x": 100}, {"x": 101}, {"x": 102}]).layout
    index = ak.layout.Index64(np.array([2, 0, 1], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)
    array = ak.Array(indexedarray)

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

    content = ak.Array([{"x": 100}, {"x": 101}, {"x": 102}]).layout
    index = ak.layout.Index64(np.array([2, 0, 1], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)
    array = ak.Array(indexedarray)

    assert f1(array, 100).tolist() == {"x": 100}
    assert f1(array, 101).tolist() == {"x": 101}
    assert f1(array, 102).tolist() == {"x": 102}
    assert f1(array, 12345) == None
