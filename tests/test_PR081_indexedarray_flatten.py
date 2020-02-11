# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_count():
    content = awkward1.layout.NumpyArray(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=numpy.int64))
    starts = awkward1.layout.Index64(numpy.array([4, 999, 0, 0, 1, 7], dtype=numpy.int64))
    stops = awkward1.layout.Index64(numpy.array([7, 999, 1, 4, 5, 10], dtype=numpy.int64))
    array1 = awkward1.layout.ListArray64(starts, stops, content)
    assert awkward1.tolist(array1) == [[4, 5, 6], [], [0], [0, 1, 2, 3], [1, 2, 3, 4], [7, 8, 9]]
    assert awkward1.tolist(array1.count()) == [3, 0, 1, 4, 4, 3]

    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    array2 = awkward1.layout.ListOffsetArray64(offsets, content)
    assert awkward1.tolist(array2) == [[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]]
    assert awkward1.tolist(array2.count()) == [3, 0, 2, 1, 4]

    array3 = awkward1.layout.RegularArray(content, 5)
    assert awkward1.tolist(array3) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert awkward1.tolist(array3.count()) == [5, 5]

    content2 = awkward1.layout.NumpyArray(numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=numpy.int64))
    assert awkward1.tolist(content2) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert awkward1.tolist(content2.count()) == [5, 5]

    index1 = awkward1.layout.Index64(numpy.array([2, 4, 0, 0, 1, 3], dtype=numpy.int64))
    array4 = awkward1.layout.IndexedArray64(index1, array2)
    assert awkward1.tolist(array4) == [[3, 4], [6, 7, 8, 9], [0, 1, 2], [0, 1, 2], [], [5]]
    assert awkward1.tolist(array4.count()) == [2, 4, 3, 3, 0, 1]

def test_indexedarray():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index1 = awkward1.layout.Index64(numpy.array([2, 3, 3, 0, 4, 8], dtype=numpy.int64))
    indexedarray1 = awkward1.layout.IndexedArray64(index1, content)
    index2 = awkward1.layout.Index64(numpy.array([2, 3, 3, -1, -1, 8], dtype=numpy.int64))
    indexedarray2 = awkward1.layout.IndexedOptionArray64(index2, content)
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6], dtype=numpy.int64))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, indexedarray1)
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets1, indexedarray2)
    offsets3 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10], dtype=numpy.int64))
    listoffsetarray3 = awkward1.layout.ListOffsetArray64(offsets3, content)
    index3 = awkward1.layout.Index64(numpy.array([2, 0, 1, 3, 3, 4], dtype=numpy.int64))
    indexedarray3 = awkward1.layout.IndexedArray64(index3, listoffsetarray3)
    index4 = awkward1.layout.Index64(numpy.array([2, -1, -1, 3, 3, 4], dtype=numpy.int64))
    indexedarray4 = awkward1.layout.IndexedOptionArray64(index4, listoffsetarray3)

    assert awkward1.tolist(indexedarray1) == [2.2, 3.3, 3.3, 0.0, 4.4, 8.8]
    assert awkward1.tolist(indexedarray2) == [2.2, 3.3, 3.3, None, None, 8.8]
    assert awkward1.tolist(listoffsetarray1) == [[2.2, 3.3, 3.3], [], [0.0, 4.4], [8.8]]
    assert awkward1.tolist(listoffsetarray2) == [[2.2, 3.3, 3.3], [], [None, None], [8.8]]
    assert awkward1.tolist(listoffsetarray3) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(indexedarray3) == [[3.3, 4.4], [0.0, 1.1, 2.2], [], [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]
    assert awkward1.tolist(indexedarray4) == [[3.3, 4.4], None, None, [5.5], [5.5], [6.6, 7.7, 8.8, 9.9]]

    with pytest.raises(ValueError) as err:
        indexedarray1.flatten()
    assert str(err.value) == "NumpyArray cannot be flattened because it has 1 dimensions"

    with pytest.raises(ValueError) as err:
        indexedarray2.flatten()
    assert str(err.value) == "NumpyArray cannot be flattened because it has 1 dimensions"

    assert awkward1.tolist(indexedarray3.flatten()) == [3.3, 4.4, 0.0, 1.1, 2.2, 5.5, 5.5, 6.6, 7.7, 8.8, 9.9]

    assert awkward1.tolist(indexedarray4.flatten()) == [3.3, 4.4, 5.5, 5.5, 6.6, 7.7, 8.8, 9.9]
