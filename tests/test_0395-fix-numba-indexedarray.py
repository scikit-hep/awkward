# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1

numba = pytest.importorskip("numba")


def test():
    def reproduce(arrays):
        out = numpy.zeros(len(arrays), numpy.int64)
        i = 0
        for values in arrays:
            for p in values:
                out[i] = p
                i += 1
                break
        return out

    numpyarray = awkward1.layout.NumpyArray(numpy.arange(100, 200, 10))[3:]
    indexedarray = awkward1.layout.IndexedArray64(awkward1.layout.Index64(numpy.array([5, 4, 3, 2, 1, 0])), numpyarray)[2:]
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 1, 4])), indexedarray)
    array = awkward1.Array(listoffsetarray)

    assert reproduce(array).tolist() == [160, 150]
    assert numba.njit(reproduce)(array).tolist() == [160, 150]

    numpyarray = awkward1.layout.NumpyArray(numpy.arange(100, 200, 10))[3:]
    indexedarray = awkward1.layout.IndexedArray64(awkward1.layout.Index64(numpy.array([5, 4, 3, 2, 1, 0])), numpyarray)[2:]
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 2, 4])), indexedarray)
    array = awkward1.Array(listoffsetarray)

    assert reproduce(array).tolist() == [160, 140]
    assert numba.njit(reproduce)(array).tolist() == [160, 140]
