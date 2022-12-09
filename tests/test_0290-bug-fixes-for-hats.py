# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_unmasked():
    @numba.njit
    def find_it(array):
        for item in array:
            if item is None:
                pass
            elif item.x == 3:
                return item
        return None

    content = ak.highlevel.Array([{"x": 1}, {"x": 2}, {"x": 3}]).layout
    unmasked = ak.contents.UnmaskedArray(content)
    array = ak.highlevel.Array(unmasked)
    assert ak.operations.to_list(find_it(array)) == {"x": 3}


def test_indexedoption():
    @numba.njit
    def find_it(array):
        for item in array:
            if item is None:
                pass
            elif item.x == 3:
                return item
        return None

    array = ak.highlevel.Array([{"x": 1}, {"x": 2}, None, {"x": 3}])
    assert ak.operations.to_list(find_it(array)) == {"x": 3}


def test_indexed_1():
    @numba.njit
    def f1(array, check):
        for i in range(len(array)):
            item = array[i]
            if item.x == check:
                return i
        return 999

    content = ak.highlevel.Array([{"x": 100}, {"x": 101}, {"x": 102}]).layout
    index = ak.index.Index64(np.array([2, 0, 1], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)
    array = ak.highlevel.Array(indexedarray)

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

    content = ak.highlevel.Array([{"x": 100}, {"x": 101}, {"x": 102}]).layout
    index = ak.index.Index64(np.array([2, 0, 1], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)
    array = ak.highlevel.Array(indexedarray)

    assert f1(array, 100).to_list() == {"x": 100}
    assert f1(array, 101).to_list() == {"x": 101}
    assert f1(array, 102).to_list() == {"x": 102}
    assert f1(array, 12345) is None
