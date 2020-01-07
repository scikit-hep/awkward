# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_basic():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int32)
    index = awkward1.layout.Index32(ind)
    array = awkward1.layout.IndexedArray32(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.uint32)
    index = awkward1.layout.IndexU32(ind)
    array = awkward1.layout.IndexedArrayU32(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int64)
    index = awkward1.layout.Index64(ind)
    array = awkward1.layout.IndexedArray64(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int32)
    index = awkward1.layout.Index32(ind)
    array = awkward1.layout.IndexedOptionArray32(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = numpy.array([2, 2, 0, 3, 4], dtype=numpy.int64)
    index = awkward1.layout.Index64(ind)
    array = awkward1.layout.IndexedOptionArray64(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 1.1, 4.4]
