# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_error():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index64(numpy.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)

    with pytest.raises(ValueError) as err:
        indexedarray.setidentities()
    assert str(err.value).startswith("in IndexedArray64 attempting to get 10, max(index) > len(content)")

def test_passthrough_32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int32))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    assert awkward1.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert numpy.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, awkward1.layout.Identities32)

def test_passthrough_U32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.IndexU32(numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.uint32))
    indexedarray = awkward1.layout.IndexedArrayU32(index, content)

    assert awkward1.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert numpy.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, awkward1.layout.Identities64)

def test_passthrough_64():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index64(numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)

    assert awkward1.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert numpy.asarray(indexedarray.content.identities).tolist() == [[0], [-1], [1], [-1], [2], [7], [3], [6], [4], [5]]
    assert isinstance(indexedarray.content.identities, awkward1.layout.Identities64)

def test_dontpass_32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(numpy.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=numpy.int32))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    assert awkward1.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None

def test_dontpass_U32():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.IndexU32(numpy.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=numpy.uint32))
    indexedarray = awkward1.layout.IndexedArrayU32(index, content)

    assert awkward1.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None

def test_dontpass_64():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index64(numpy.array([0, 2, 4, 6, 8, 6, 4, 2, 0], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)

    assert awkward1.to_list(indexedarray) == [0.0, 2.2, 4.4, 6.6, 8.8, 6.6, 4.4, 2.2, 0.0]
    indexedarray.setidentities()
    assert numpy.asarray(indexedarray.identities).tolist() == [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    assert indexedarray.content.identities is None
