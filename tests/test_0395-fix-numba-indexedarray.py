# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak

numba = pytest.importorskip("numba")


def test():
    def reproduce(arrays):
        out = np.zeros(len(arrays), np.int64)
        i = 0
        for values in arrays:
            for p in values:
                out[i] = p
                i += 1
                break
        return out

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 1, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [150, 140]
    assert numba.njit(reproduce)(array).tolist() == [150, 140]

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 2, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [150, 130]
    assert numba.njit(reproduce)(array).tolist() == [150, 130]

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )[2:]
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 1, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [130, 120]
    assert numba.njit(reproduce)(array).tolist() == [130, 120]

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )[2:]
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 2, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [130, 110]
    assert numba.njit(reproduce)(array).tolist() == [130, 110]

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))[3:]
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 1, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [180, 170]
    assert numba.njit(reproduce)(array).tolist() == [180, 170]

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))[3:]
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 2, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [180, 160]
    assert numba.njit(reproduce)(array).tolist() == [180, 160]

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))[3:]
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )[2:]
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 1, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [160, 150]
    assert numba.njit(reproduce)(array).tolist() == [160, 150]

    numpyarray = ak.layout.NumpyArray(np.arange(100, 200, 10))[3:]
    indexedarray = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([5, 4, 3, 2, 1, 0])), numpyarray
    )[2:]
    listoffsetarray = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 2, 4])), indexedarray
    )
    array = ak.Array(listoffsetarray)

    assert reproduce(array).tolist() == [160, 140]
    assert numba.njit(reproduce)(array).tolist() == [160, 140]


def test_enumerate():
    @numba.njit
    def f1(array):
        out = np.zeros(len(array), np.int32)
        for i, x in enumerate(array):
            out[i] = x + 0.5
        return out

    array = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    assert f1(array).tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]


def test_zip():
    @numba.njit
    def f1(array1, array2):
        out = np.zeros(len(array1), np.float64)
        i = 0
        for a1, a2 in zip(array1, array2):
            out[i] = a1 - a2
            i += 1
        return out

    array1 = ak.Array([1.5, 2.5, 3.25])
    array2 = ak.Array([1, 2, 3])
    assert f1(array1, array2).tolist() == [0.5, 0.5, 0.25]
