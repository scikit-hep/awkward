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

def test_type():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = awkward1.layout.Index32(numpy.array([2, 2, 0, 3, 4], dtype=numpy.int32))
    array = awkward1.layout.IndexedArray32(index, content)
    assert awkward1.typeof(array) == awkward1.layout.PrimitiveType("float64")
    array = awkward1.layout.IndexedOptionArray32(index, content)
    assert awkward1.typeof(array) == awkward1.layout.OptionType(awkward1.layout.PrimitiveType("float64"))

def test_null():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = awkward1.layout.Index64(numpy.array([2, 2, 0, -1, 4], dtype=numpy.int64))
    array = awkward1.layout.IndexedOptionArray64(index, content)
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, None, 4.4]

def test_carry():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = awkward1.layout.Index64(numpy.array([2, 2, 0, 3, 4], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, indexedarray)
    assert awkward1.tolist(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, 4.4]]
    assert awkward1.tolist(listoffsetarray[::-1]) == [[3.3, 4.4], [], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0]]) == [[3.3, 4.4], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0], 1]) == [4.4, 2.2]     # invokes carry
    assert awkward1.tolist(listoffsetarray[2:, 1]) == [4.4]              # invokes carry

    index = awkward1.layout.Index64(numpy.array([2, 2, 0, 3, -1], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedOptionArray64(index, content)
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, indexedarray)
    assert awkward1.tolist(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, None]]
    assert awkward1.tolist(listoffsetarray[::-1]) == [[3.3, None], [], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0]]) == [[3.3, None], [2.2, 2.2, 0.0]]
    assert awkward1.tolist(listoffsetarray[[2, 0], 1]) == [None, 2.2]    # invokes carry
    assert awkward1.tolist(listoffsetarray[2:, 1]) == [None]             # invokes carry

def test_others():
    content = awkward1.layout.NumpyArray(numpy.array([[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]))
    index = awkward1.layout.Index64(numpy.array([4, 0, 3, 1, 3], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    assert indexedarray[3, 0] == 0.1
    assert indexedarray[3, 1] == 1.0
    assert awkward1.tolist(indexedarray[3, ::-1]) == [1.0, 0.1]
    assert awkward1.tolist(indexedarray[3, [1, 1, 0]]) == [1.0, 1.0, 0.1]
    assert awkward1.tolist(indexedarray[3:, 0]) == [0.1, 0.3]
    assert awkward1.tolist(indexedarray[3:, 1]) == [1.0, 3.0]
    assert awkward1.tolist(indexedarray[3:, ::-1]) == [[1.0, 0.1], [3.0, 0.3]]
    assert awkward1.tolist(indexedarray[3:, [1, 1, 0]]) == [[1.0, 1.0, 0.1], [3.0, 3.0, 0.3]]

def test_missing():
    content = awkward1.layout.NumpyArray(numpy.array([[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]))
    index = awkward1.layout.Index64(numpy.array([4, 0, 3, -1, 3], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedOptionArray64(index, content)
    assert awkward1.tolist(indexedarray[3:, 0]) == [None, 0.3]
    assert awkward1.tolist(indexedarray[3:, 1]) == [None, 3.0]
    assert awkward1.tolist(indexedarray[3:, ::-1]) == [None, [3.0, 0.3]]
    assert awkward1.tolist(indexedarray[3:, [1, 1, 0]]) == [None, [3.0, 3.0, 0.3]]

def test_highlevel():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = awkward1.layout.Index64(numpy.array([2, 2, 0, -1, 4], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.IndexedOptionArray64(index, content))
    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, None, 4.4]
    assert str(array) == "[2.2, 2.2, 0, None, 4.4]"
    assert repr(array) == "<Array [2.2, 2.2, 0, None, 4.4] type='5 * ?float64'>"

def test_fillable():
    assert awkward1.tolist(awkward1.Array([1.1, 2.2, 3.3, None, 4.4])) == [1.1, 2.2, 3.3, None, 4.4]
    assert awkward1.tolist(awkward1.Array([None, 2.2, 3.3, None, 4.4])) == [None, 2.2, 3.3, None, 4.4]

    assert awkward1.tolist(awkward1.Array([[1.1, 2.2, 3.3], [], [None, 4.4]])) == [[1.1, 2.2, 3.3], [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([[1.1, 2.2, 3.3], [], None, [None, 4.4]])) == [[1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([[1.1, 2.2, 3.3], None, [], [None, 4.4]])) == [[1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([[1.1, 2.2, 3.3], None, [], [None, 4.4]])) != [[1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert awkward1.tolist(awkward1.Array([[None, 1.1, 2.2, 3.3], [], [None, 4.4]])) == [[None, 1.1, 2.2, 3.3], [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]])) == [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]])) == [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]])) != [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert awkward1.tolist(awkward1.Array([None, [1.1, 2.2, 3.3], [], [None, 4.4]])) == [None, [1.1, 2.2, 3.3], [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([None, [1.1, 2.2, 3.3], [], None, [None, 4.4]])) == [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([None, [1.1, 2.2, 3.3], None, [], [None, 4.4]])) == [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array([None, [1.1, 2.2, 3.3], None, [], [None, 4.4]])) != [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert awkward1.tolist(awkward1.Array([None, None, None, None, None])) == [None, None, None, None, None]
    assert awkward1.tolist(awkward1.Array([[None, None, None], [], [None, None]])) == [[None, None, None], [], [None, None]]

def test_json():
    assert awkward1.tolist(awkward1.Array("[1.1, 2.2, 3.3, null, 4.4]")) == [1.1, 2.2, 3.3, None, 4.4]
    assert awkward1.tolist(awkward1.Array("[null, 2.2, 3.3, null, 4.4]")) == [None, 2.2, 3.3, None, 4.4]

    assert awkward1.tolist(awkward1.Array("[[1.1, 2.2, 3.3], [], [null, 4.4]]")) == [[1.1, 2.2, 3.3], [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[[1.1, 2.2, 3.3], [], null, [null, 4.4]]")) == [[1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[[1.1, 2.2, 3.3], null, [], [null, 4.4]]")) == [[1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[[1.1, 2.2, 3.3], null, [], [null, 4.4]]")) != [[1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert awkward1.tolist(awkward1.Array("[[null, 1.1, 2.2, 3.3], [], [null, 4.4]]")) == [[None, 1.1, 2.2, 3.3], [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[[null, 1.1, 2.2, 3.3], [], null, [null, 4.4]]")) == [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[[null, 1.1, 2.2, 3.3], null, [], [null, 4.4]]")) == [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[[null, 1.1, 2.2, 3.3], null, [], [null, 4.4]]")) != [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert awkward1.tolist(awkward1.Array("[null, [1.1, 2.2, 3.3], [], [null, 4.4]]")) == [None, [1.1, 2.2, 3.3], [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[null, [1.1, 2.2, 3.3], [], null, [null, 4.4]]")) == [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[null, [1.1, 2.2, 3.3], null, [], [null, 4.4]]")) == [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert awkward1.tolist(awkward1.Array("[null, [1.1, 2.2, 3.3], null, [], [null, 4.4]]")) != [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert awkward1.tolist(awkward1.Array("[null, null, null, null, null]")) == [None, None, None, None, None]
    assert awkward1.tolist(awkward1.Array("[[null, null, null], [], [null, null]]")) == [[None, None, None], [], [None, None]]

def test_emptyarray():
    array = awkward1.layout.EmptyArray()
    assert isinstance(array.astype(awkward1.layout.OptionType(awkward1.layout.PrimitiveType("float32"))), awkward1.layout.IndexedOptionArray64)

    offsets = awkward1.layout.Index64(numpy.array([0], dtype=numpy.int64))
    array = awkward1.layout.ListOffsetArray64(offsets, awkward1.layout.EmptyArray())
    assert isinstance(array.astype(awkward1.layout.ListType(awkward1.layout.OptionType(awkward1.layout.PrimitiveType("float32")))).content, awkward1.layout.IndexedOptionArray64)

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

    # assert awkward1.tolist(indexedarray4.flatten()) == [3.3, 4.4, None, None, 5.5, 5.5, 6.6, 7.7, 8.8, 9.9]
