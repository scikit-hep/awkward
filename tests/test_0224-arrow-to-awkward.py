# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

pyarrow = pytest.importorskip("pyarrow")

def test_toarrow_BitMaskedArray():
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bitmask = awkward1.layout.IndexU8(numpy.array([40, 34], dtype=numpy.uint8))
    array = awkward1.layout.BitMaskedArray(bitmask, content, False, 9, False)
    assert awkward1.to_arrow(array).to_pylist() == awkward1.to_list(array)

def test_toarrow_ByteMaskedArray_1():
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    bytemask = awkward1.layout.Index8(
        numpy.array([False, True, False], dtype=numpy.bool))
    array = awkward1.layout.ByteMaskedArray(bytemask, content, True)
    assert awkward1.to_arrow(array).to_pylist() == awkward1.to_list(array)

def test_toarrow_NumpyArray_1():
    array = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    assert isinstance(awkward1.to_arrow(array),
                      (pyarrow.lib.Tensor, pyarrow.lib.Array))
    assert awkward1.to_arrow(array).to_pylist() == [
        0.0, 1.1, 2.2, 3.3, 4.4, 5.5]

def test_toarrow_NumpyArray_2():
    array = awkward1.layout.NumpyArray(
        numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]))
    assert isinstance(awkward1.to_arrow(array),
                      (pyarrow.lib.Tensor, pyarrow.lib.Array))
    assert awkward1.to_arrow(array) == pyarrow.Tensor.from_numpy(
        numpy.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]))

def test_toarrow_EmptyArray():
    array = awkward1.layout.EmptyArray()
    assert isinstance(awkward1.to_arrow(array), (pyarrow.lib.Array))
    assert awkward1.to_arrow(array).to_pylist() == []

def test_toarrow_ListOffsetArray64():
    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArray64(offsets, content)
    assert isinstance(awkward1.to_arrow(array), (pyarrow.ListArray))
    assert awkward1.to_arrow(array).to_pylist() == [[1.1, 2.2, 3.3], [], [
        4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

def test_toarrow_ListOffsetArrayU32():
    content = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.IndexU32(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArrayU32(offsets, content)
    assert isinstance(awkward1.to_arrow(array), (pyarrow.ListArray))
    assert awkward1.to_arrow(array).to_pylist() == [[1.1, 2.2, 3.3], [], [
        4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

def test_toarrow_ListArray_RegularArray():
    # Testing parameters
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert awkward1.to_arrow(array).to_pylist() == [['one', 'two', 'three'], [], [
        'four', 'five'], ['six'], ['seven', 'eight', 'nine']]

    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

    assert isinstance(awkward1.to_arrow(listarray), (pyarrow.ListArray))
    assert awkward1.to_arrow(listarray).to_pylist() == [[[[0.0, 1.1, 2.2], []], [
        [3.3, 4.4], [5.5]]], [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]]

    assert isinstance(awkward1.to_arrow(regulararray), (pyarrow.ListArray))
    assert awkward1.to_arrow(regulararray).to_pylist() == [[[0.0, 1.1, 2.2], []], [
        [3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]

def test_toarrow_RecordArray():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))

    recordarray = awkward1.layout.RecordArray(
        [content1, listoffsetarray, content2, content1], keys=["one", "two", "2", "wonky"])

    assert isinstance(awkward1.to_arrow(recordarray), (pyarrow.StructArray))
    assert awkward1.to_arrow(recordarray).to_pylist() == [{'one': 1, 'two': [0.0, 1.1, 2.2], '2': 1.1, 'wonky': 1}, {'one': 2, 'two': [], '2': 2.2, 'wonky': 2}, {
        'one': 3, 'two': [3.3, 4.4], '2': 3.3, 'wonky': 3}, {'one': 4, 'two': [5.5], '2': 4.4, 'wonky': 4}, {'one': 5, 'two': [6.6, 7.7, 8.8, 9.9], '2': 5.5, 'wonky': 5}]

def test_toarrow_UnionArray():
    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    tags = awkward1.layout.Index8(
        numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(
        numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    unionarray = awkward1.layout.UnionArray8_32(
        tags, index, [content0, content1])

    assert isinstance(awkward1.to_arrow(unionarray), (pyarrow.UnionArray))
    assert awkward1.to_arrow(unionarray).to_pylist() == [
        1, 2, [1.1, 2.2, 3.3], [], 3, [4.4, 5.5], 5, 4]

def test_toarrow_IndexedArray():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(
        numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    assert isinstance(awkward1.to_arrow(indexedarray),
                      (pyarrow.DictionaryArray))
    assert awkward1.to_arrow(indexedarray).to_pylist() == [
        0.0, 2.2, 4.4, 6.6, 8.8, 9.9, 7.7, 5.5]

def test_toarrow_ByteMaskedArray_2():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array(
        [True, True, False, False, False], dtype=numpy.int8)), listoffsetarray, True)

    assert awkward1.to_arrow(bytemaskedarray).to_pylist() == [
        [0.0, 1.1, 2.2], [], None, None, None]

def test_toarrow_ByteMaskedArray_3():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    starts = awkward1.layout.Index64(numpy.array([0, 1]))
    stops = awkward1.layout.Index64(numpy.array([2, 3]))
    listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False], dtype=numpy.int8)), listarray, True)
    assert awkward1.to_arrow(bytemaskedarray).to_pylist(
    ) == awkward1.to_list(bytemaskedarray)

def test_toarrow_ByteMaskedArray_4():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    recordarray = awkward1.layout.RecordArray(
        [content1, listoffsetarray, content2, content1], keys=["one", "two", "2", "wonky"])

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False], dtype=numpy.int8)), recordarray, True)
    assert awkward1.to_arrow(bytemaskedarray).to_pylist(
    ) == awkward1.to_list(bytemaskedarray)

def test_toarrow_ByteMaskedArray_5():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    index = awkward1.layout.Index32(
        numpy.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray32(index, content)

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([True, False, False], dtype=numpy.int8)), indexedarray, True)
    assert awkward1.to_arrow(bytemaskedarray).to_pylist() == awkward1.to_list(bytemaskedarray)

@pytest.mark.skip(reason="https://issues.apache.org/jira/browse/ARROW-9556")
def test_toarrow_ByteMaskedArray_broken_unions():
    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    tags = awkward1.layout.Index8(
        numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(
        numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    unionarray = awkward1.layout.UnionArray8_32(
        tags, index, [content0, content1])

    bytemaskedarray = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(
        numpy.array([True, False, False], dtype=numpy.int8)), unionarray, True)
    assert awkward1.to_arrow(bytemaskedarray).to_pylist(
    ) == awkward1.to_list(bytemaskedarray)

def test_toarrow_IndexedOptionArray():
    ioa = awkward1.layout.IndexedOptionArray32(
        awkward1.layout.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
        awkward1.layout.NumpyArray(numpy.array(
            [5.2, 1.7, 6.7, -0.4, 4.0, 7.8, 3.8, 6.8, 4.2, 0.3, 4.6, 6.2,
             6.9, -0.7, 3.9, 1.6, 8.7, -0.7, 3.2, 4.3, 4.0, 5.8, 4.2, 7.0,
             5.6, 3.8])))
    assert awkward1.to_arrow(ioa).to_pylist() == awkward1.to_list(ioa)

def test_fromarrow_NumpyArray_1():
    boolarray = awkward1.layout.NumpyArray(numpy.array([True, True, True, False, False, True, False, True, False, True]))
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(boolarray), highlevel=False)) == awkward1.to_list(boolarray)

def test_fromarrow_NumpyArray_2():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(content), highlevel=False)) == awkward1.to_list(content)

def test_fromarrow_ListOffsetArray():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(listoffsetarray), highlevel=False)) == awkward1.to_list(listoffsetarray)

def test_fromarrow_RegularArray():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)

    regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(regulararray), highlevel=False)) == awkward1.to_list(regulararray)

def test_fromarrow_RecordArray():
    content = awkward1.layout.NumpyArray(
        numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)

    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(
        numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index32(numpy.array([0, 3, 3, 5, 6, 9]))
    recordarray = awkward1.layout.RecordArray(
        [content1, listoffsetarray, content2, content1], keys=["one", "chonks", "2", "wonky"])
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(recordarray), highlevel=False)) == awkward1.to_list(recordarray)

def test_fromarrow_UnionArray():
    content0 = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = awkward1.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]).layout
    tags = awkward1.layout.Index8(
        numpy.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=numpy.int8))
    index = awkward1.layout.Index32(
        numpy.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=numpy.int32))
    array = awkward1.layout.UnionArray8_32(tags, index, [content0, content])
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(array), highlevel=False)) == awkward1.to_list(array)

def test_chunkedarray():
    a = pyarrow.chunked_array([pyarrow.array([1.1, 2.2, 3.3]), pyarrow.array([], pyarrow.float64()), pyarrow.array([4.4, 5.5]), pyarrow.array([6.6]), pyarrow.array([], pyarrow.float64()), pyarrow.array([], pyarrow.float64()), pyarrow.array([7.7, 8.8, 9.9])])
    assert a.to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

def test_recordbatch():
    a = pyarrow.RecordBatch.from_arrays(
            [pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
             pyarrow.array([[1, 2, 3], [], [], [4, 5], [6]])],
            ["a", "b"])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"a": 1.1, "b": [1, 2, 3]}, {"a": 2.2, "b": []}, {"a": 3.3, "b": []}, {"a": 4.4, "b": [4, 5]}, {"a": 5.5, "b": [6]}]

    a = pyarrow.RecordBatch.from_arrays(
            [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
             pyarrow.array([[1, None, 3], [], [], [4, 5], [6]])],
            ["a", "b"])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"a": 1.1, "b": [1, None, 3]}, {"a": 2.2, "b": []}, {"a": 3.3, "b": []}, {"a": None, "b": [4, 5]}, {"a": 5.5, "b": [6]}]

    a = pyarrow.RecordBatch.from_arrays(
            [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
             pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
             pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
             pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
             pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
            ["a", "b", "c", "d", "e"])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"a": 1.1, "b": [1, 2, 3], "c": {"x": 1, "y": 1.1}, "d": {"x": 1, "y": 1.1}, "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]}, {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []}, {"a": 3.3, "b": [4, 5], "c": {"x": 3, "y": 3.3}, "d": None, "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}]}, {"a": None, "b": [None], "c": {"x": 4, "y": None}, "d":{"x": 4, "y": None}, "e": [None]}, {"a": 5.5, "b": [6], "c": {"x": 5, "y": 5.5}, "d": {"x": 5, "y": 5.5}, "e": [{"x": 6, "y": 6.6}]}]

### All of the following tests were copied (translated) over from Awkward 0.

def test_arrow_toarrow_string():
    a = awkward1.from_iter(["one", "two", "three"], highlevel=False)
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a), highlevel=False)) == awkward1.to_list(a)
    a = awkward1.from_iter([["one", "two", "three"], [], ["four", "five"]], highlevel=False)
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a), highlevel=False)) == awkward1.to_list(a)
    if hasattr(pyarrow.BinaryArray, 'from_buffers'):
        a = awkward1.from_iter([b"one", b"two", b"three"], highlevel=False)
        assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a), highlevel=False)) == [b"one", b"two", b"three"]
        a = awkward1.from_iter([[b"one", b"two", b"three"], [], [b"four", b"five"]], highlevel=False)
        assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a), highlevel=False)) == [[b"one", b"two", b"three"], [], [b"four", b"five"]]
    else:
        a = awkward1.from_iter([b"one", b"two", b"three"], highlevel=False)
        assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a), highlevel=False)) == ["one", "two", "three"]
        a = awkward1.from_iter([[b"one", b"two", b"three"], [], [b"four", b"five"]], highlevel=False)
        assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a), highlevel=False)) == [["one", "two", "three"], [], ["four", "five"]]

def test_arrow_array():
    a = pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [1.1, 2.2, 3.3, 4.4, 5.5]

def test_arrow_boolean():
    a = pyarrow.array([True, True, False, False, True])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [True, True, False, False, True]

def test_arrow_array_null():
    a = pyarrow.array([1.1, 2.2, 3.3, None, 4.4, 5.5])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [1.1, 2.2, 3.3, None, 4.4, 5.5]

def test_arrow_nested_array():
    a = pyarrow.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

def test_arrow_nested_nested_array():
    a = pyarrow.array([[[1.1, 2.2], [3.3], []], [], [[4.4, 5.5]]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[[1.1, 2.2], [3.3], []], [], [[4.4, 5.5]]]

def test_arrow_nested_array_null():
    a = pyarrow.array([[1.1, 2.2, None], [], [4.4, 5.5]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[1.1, 2.2, None], [], [4.4, 5.5]]

def test_arrow_null_nested_array_null():
    a = pyarrow.array([[1.1, 2.2, None], [], None, [4.4, 5.5]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[1.1, 2.2, None], [], None, [4.4, 5.5]]

def test_arrow_chunked_array():
    a = pyarrow.chunked_array([pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]), pyarrow.array([], pyarrow.float64()), pyarrow.array([6.6, 7.7, 8.8])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]

def test_arrow_struct():
    a = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

def test_arrow_struct_null():
    a = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}]

def test_arrow_null_struct():
    a = pyarrow.array([{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

def test_arrow_null_struct_null():
    a = pyarrow.array([{"x": 1, "y": 1.1}, None, {"x": 2, "y": None}, {"x": 3, "y": 3.3}])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"x": 1, "y": 1.1}, None, {"x": 2, "y": None}, {"x": 3, "y": 3.3}]

def test_arrow_chunked_struct():
    t = pyarrow.struct({"x": pyarrow.int64(), "y": pyarrow.float64()})
    a = pyarrow.chunked_array([pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], t), pyarrow.array([], t), pyarrow.array([{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}], t)])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]

def test_arrow_nested_struct():
    a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

def test_arrow_nested_struct_null():
    a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

def test_arrow_null_nested_struct():
    a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

def test_arrow_null_nested_struct_null():
    a = pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}], None, [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

def test_arrow_struct_nested():
    a = pyarrow.array([{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}]

def test_arrow_struct_nested_null():
    a = pyarrow.array([{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}]

def test_arrow_nested_struct_nested():
    a = pyarrow.array([[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}], [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}], [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]]

def test_arrow_null_nested_struct_nested_null():
    a = pyarrow.array([[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}], None, [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [[{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}], None, [], [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}]]

def test_arrow_strings():
    if not awkward1._util.py27:
        a = pyarrow.array(["one", "two", "three", u"fo\u2014ur", "five"])
        assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == ["one", "two", "three", u"fo\u2014ur", "five"]

def test_arrow_strings_null():
    if not awkward1._util.py27:
        a = pyarrow.array(["one", "two", None, u"fo\u2014ur", "five"])
        assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == ["one", "two", None, u"fo\u2014ur", "five"]

def test_arrow_binary():
    a = pyarrow.array([b"one", b"two", b"three", b"four", b"five"])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [b"one", b"two", b"three", b"four", b"five"]

def test_arrow_binary_null():
    a = pyarrow.array([b"one", b"two", None, b"four", b"five"])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [b"one", b"two", None, b"four", b"five"]

def test_arrow_chunked_strings():
    a = pyarrow.chunked_array([pyarrow.array(["one", "two", "three", "four", "five"]), pyarrow.array(["six", "seven", "eight"])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == ["one", "two", "three", "four", "five", "six", "seven", "eight"]

def test_arrow_nested_strings():
    a = pyarrow.array([["one", "two", "three"], [], ["four", "five"]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [["one", "two", "three"], [], ["four", "five"]]

def test_arrow_nested_strings_null():
    a = pyarrow.array([["one", "two", None], [], ["four", "five"]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [["one", "two", None], [], ["four", "five"]]

def test_arrow_null_nested_strings_null():
    a = pyarrow.array([["one", "two", None], [], None, ["four", "five"]])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [["one", "two", None], [], None, ["four", "five"]]

def test_arrow_union_sparse():
    a = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, 2.2, 3.3, 4.4]), pyarrow.array([True, True, False, True, False])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [0.0, True, 2.2, 3.3, False]

def test_arrow_union_sparse_null():
    a = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, None, 3.3, 4.4]), pyarrow.array([True, True, False, True, False])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [0.0, True, None, 3.3, False]

def test_arrow_union_sparse_null_null():
    a = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, None, 3.3, 4.4]), pyarrow.array([True, None, False, True, False])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [0.0, None, None, 3.3, False]

def test_arrow_union_dense():
    a = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, 2.2, 3.3]), pyarrow.array([True, True, False])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [0.0, True, 1.1, 2.2, 3.3, True, False]

def test_arrow_union_dense_null():
    a = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, True, False])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [0.0, True, 1.1, None, 3.3, True, False]

def test_arrow_union_dense_null_null():
    a = pyarrow.UnionArray.from_dense(pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()), pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()), [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, None, False])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [0.0, True, 1.1, None, 3.3, None, False]

def test_arrow_dictarray():
    a = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]), pyarrow.array(["one", "two", "three"]))
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == ["one", "one", "three", "three", "two", "one", "three", "two", "two"]

def test_arrow_dictarray_null():
    a = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, None, 1, None, 2, 1, 1]), pyarrow.array(["one", "two", "three"]))
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == ["one", "one", "three", None, "two", None, "three", "two", "two"]

def test_arrow_null_dictarray():
    a = pyarrow.DictionaryArray.from_arrays(pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]), pyarrow.array(["one", None, "three"]))
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == ["one", "one", "three", "three", None, "one", "three", None, None]

def test_arrow_batch():
    a = pyarrow.RecordBatch.from_arrays(
        [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
         pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
         pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
         pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
         pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
        ["a", "b", "c", "d", "e"])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"a": 1.1, "b": [1, 2, 3], "c": {"x": 1, "y": 1.1}, "d": {"x": 1, "y": 1.1}, "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]}, {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []}, {"a": 3.3, "b": [4, 5], "c": {"x": 3, "y": 3.3}, "d": None, "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}]}, {"a": None, "b": [None], "c": {"x": 4, "y": None}, "d":{"x": 4, "y": None}, "e": [None]}, {"a": 5.5, "b": [6], "c": {"x": 5, "y": 5.5}, "d": {"x": 5, "y": 5.5}, "e": [{"x": 6, "y": 6.6}]}]

def test_arrow_table():
    a = pyarrow.Table.from_batches([
        pyarrow.RecordBatch.from_arrays(
        [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
         pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
         pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
         pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
         pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
        ["a", "b", "c", "d", "e"]),
        pyarrow.RecordBatch.from_arrays(
        [pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
         pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
         pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
         pyarrow.array([{"x": 1, "y": 1.1}, None, None, {"x": 4, "y": None}, {"x": 5, "y": 5.5}]),
         pyarrow.array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": None}, {"x": 5, "y": 5.5}], [None], [{"x": 6, "y": 6.6}]])],
        ["a", "b", "c", "d", "e"])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [{"a": 1.1, "b": [1, 2, 3], "c": {"x": 1, "y": 1.1}, "d": {"x": 1, "y": 1.1}, "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]}, {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []}, {"a": 3.3, "b": [4, 5], "c": {"x": 3, "y": 3.3}, "d": None, "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}]}, {"a": None, "b": [None], "c": {"x": 4, "y": None}, "d": {"x": 4, "y": None}, "e": [None]}, {"a": 5.5, "b": [6], "c": {"x": 5, "y": 5.5}, "d": {"x": 5, "y": 5.5}, "e": [{"x": 6, "y": 6.6}]}, {"a": 1.1, "b": [1, 2, 3], "c": {"x": 1, "y": 1.1}, "d": {"x": 1, "y": 1.1}, "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]}, {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []}, {"a": 3.3, "b": [4, 5], "c": {"x": 3, "y": 3.3}, "d": None, "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}]}, {"a": None, "b": [None], "c": {"x": 4, "y": None}, "d": {"x": 4, "y": None}, "e": [None]}, {"a": 5.5, "b": [6], "c": {"x": 5, "y": 5.5}, "d": {"x": 5, "y": 5.5}, "e": [{"x": 6, "y": 6.6}]}]

def test_arrow_nonnullable_table():
    x = pyarrow.array([1, 2, 3])
    y = pyarrow.array([1.1, 2.2, 3.3])
    table = pyarrow.Table.from_arrays([x], ["x"])
    if hasattr(pyarrow, "column"):
        table2 = table.add_column(1, pyarrow.column(pyarrow.field("y", y.type, False), numpy.array([1.1, 2.2, 3.3])))
    else:
        table2 = table.add_column(1, "y", y)
    assert awkward1.to_list(awkward1.from_arrow(table2, highlevel=False)) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

def test_arrow_coverage100():
    a = awkward1.from_iter([True, True, False, False, True, False, True, False], highlevel=False)
    assert awkward1.to_arrow(a).to_pylist() == awkward1.to_list(a)

    a = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"), parameters={"__array__": "bytes"}), parameters={"__array__": "bytestring"})
    assert awkward1.to_arrow(a).to_pylist() == [b"hello", b"there"]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True, False, False, True, True])), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10, 15, 20, 25, 30], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellotherehellotherehellothere", "u1"), parameters={"__array__": "bytes"}), parameters={"__array__": "bytestring"}), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == [b"hello", None, b"hello", b"there", None, None]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True])), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"), parameters={"__array__": "bytes"}), parameters={"__array__": "bytestring"}), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == [b"hello", None]

    a = awkward1.layout.IndexedOptionArray32(awkward1.layout.Index32(numpy.array([-1, 1, -1, 0, 0, -1], "i4")), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"), parameters={"__array__": "bytes"}), parameters={"__array__": "bytestring"}))
    assert awkward1.to_arrow(a).to_pylist() == [None, b"there", None, b"hello", b"hello", None]

    a = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"), parameters={"__array__": "chars"}), parameters={"__array__": "string"})
    assert awkward1.to_arrow(a).to_pylist() == ["hello", "there"]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True, False, False, True, True])), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10, 15, 20, 25, 30], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellotherehellotherehellothere", "u1"), parameters={"__array__": "chars"}), parameters={"__array__": "string"}), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == ["hello", None, "hello", "there", None, None]
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a))) == ["hello", None, "hello", "there", None, None]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True, False, False, True, True])), awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 25, 30], "i8")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellotherehellotherehellothere", "u1"), parameters={"__array__": "chars"}), parameters={"__array__": "string"}), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == ["hello", None, "hello", "there", None, None]
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a))) == ["hello", None, "hello", "there", None, None]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True, False, False, True, True])), awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 5, 10, 15, 20, 25, 30], "i8")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellotherehellotherehellothere", "u1"), parameters={"__array__": "bytes"}), parameters={"__array__": "bytestring"}), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == [b"hello", None, b"hello", b"there", None, None]
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a))) == [b"hello", None, b"hello", b"there", None, None]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True])), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"), parameters={"__array__": "chars"}), parameters={"__array__": "string"}), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == ["hello", None]

    a = awkward1.layout.IndexedOptionArray32(awkward1.layout.Index32(numpy.array([-1, 1, -1, 0, 0, -1], "i4")), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"), parameters={"__array__": "chars"}), parameters={"__array__": "string"}))
    assert awkward1.to_arrow(a).to_pylist() == [None, "there", None, "hello", "hello", None]

    a = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1")))
    assert awkward1.to_arrow(a).to_pylist() == [[104, 101, 108, 108, 111], [116, 104, 101, 114, 101]]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True, False, False, True, True])), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10, 15, 20, 25, 30], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellotherehellotherehellothere", "u1"))), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == [[104, 101, 108, 108, 111], None, [104, 101, 108, 108, 111], [116, 104, 101, 114, 101], None, None]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, True])), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"))), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == [[104, 101, 108, 108, 111], None]

    a = awkward1.layout.IndexedOptionArray32(awkward1.layout.Index32(numpy.array([-1, 1, -1, 0, 0, -1], "i4")), awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"))))
    assert awkward1.to_arrow(a).to_pylist() == [None, [116, 104, 101, 114, 101], None, [104, 101, 108, 108, 111], [104, 101, 108, 108, 111], None]

    a = awkward1.layout.IndexedOptionArray32(awkward1.layout.Index32(numpy.array([-1, 1, -1, 0, 0, -1], "i4")), awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])), 3))
    assert awkward1.to_arrow(a).to_pylist() == [None, [4.4, 5.5, 6.6], None, [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], None]

    a = awkward1.layout.IndexedOptionArray32(awkward1.layout.Index32(numpy.array([-1, 1, -1, 0, 0, -1, 1, -1], "i4")), awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])), 3))
    assert awkward1.to_arrow(a).to_pylist() == [None, [4.4, 5.5, 6.6], None, [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], None, [4.4, 5.5, 6.6], None]

    a = awkward1.layout.IndexedOptionArray64(awkward1.layout.Index64(numpy.array([-1, 1, -1, 0, 0, -1, 1, -1], "i8")), awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])), 3))
    assert awkward1.to_arrow(a).to_pylist() == [None, [4.4, 5.5, 6.6], None, [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], None, [4.4, 5.5, 6.6], None]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([True, True, True, True, False, False])), awkward1.layout.IndexedOptionArray32(awkward1.layout.Index32(numpy.array([-1, 1, -1, 0, 0, -1], "i4")), awkward1.layout.RegularArray(awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])), 3)), valid_when=True)
    assert awkward1.to_arrow(a).to_pylist() == [None, [4.4, 5.5, 6.6], None, [1.1, 2.2, 3.3], None, None]

    a = awkward1.layout.UnmaskedArray(awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(numpy.array([0, 5, 10], "i4")), awkward1.layout.NumpyArray(numpy.frombuffer(b"hellothere", "u1"))))
    assert awkward1.to_arrow(a).to_pylist() == [[104, 101, 108, 108, 111], [116, 104, 101, 114, 101]]

    a = pyarrow.array(["one", "two", "three", "two", "two", "one", "three", "one"]).dictionary_encode()
    b = awkward1.from_arrow(a, highlevel=False)
    assert isinstance(b, awkward1._util.indexedtypes)
    assert awkward1.to_list(b) == ["one", "two", "three", "two", "two", "one", "three", "one"]

    a = awkward1.Array([[1.1, 2.2, 3.3], [], None, [4.4, 5.5]])
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a))) == [[1.1, 2.2, 3.3], [], None, [4.4, 5.5]]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, False, False, True, True, False, False])), awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 999, 314, 4.4, 5.5])), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == [1.1, 2.2, 3.3, None, None, 4.4, 5.5]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([False, False, False, True, True, False, False])), awkward1.from_iter([b"hello", b"", b"there", b"yuk", b"", b"o", b"hellothere"], highlevel=False), valid_when=False)
    assert awkward1.to_arrow(a).to_pylist() == [b"hello", b"", b"there", None, None, b"o", b"hellothere"]

    a = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8([True, True, False, True]), awkward1.from_iter([[1.1, 2.2, 3.3], [], [999], [4.4, 5.5]], highlevel=False), valid_when=True)
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a))) == [[1.1, 2.2, 3.3], [], None, [4.4, 5.5]]

    a = awkward1.from_iter([[1, 2, 3], [], [4, 5], 999, 123], highlevel=False)
    assert awkward1.to_arrow(a).to_pylist() == [[1, 2, 3], [], [4, 5], 999, 123]
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a))) == [[1, 2, 3], [], [4, 5], 999, 123]

@pytest.mark.skip(reason="https://issues.apache.org/jira/browse/ARROW-9556")
def test_arrow_coverage100_broken_unions():
    a = awkward1.from_iter([[1, 2, 3], [], [4, 5], 999, 123], highlevel=False)
    b = awkward1.layout.ByteMaskedArray(awkward1.layout.Index8(numpy.array([True, True, False, False, True])), a, valid_when=True)
    assert awkward1.to_arrow(b).to_pylist() == [[1, 2, 3], [], None, None, 123]
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(b))) == [[1, 2, 3], [], None, None, 123]

    content1 = awkward1.from_iter([1.1, 2.2, 3.3, 4.4, 5.5], highlevel=False)
    content2 = awkward1.layout.NumpyArray(numpy.array([], dtype=numpy.int32))
    a = awkward1.layout.UnionArray8_32(awkward1.layout.Index8(numpy.array([0, 0, 0, 0, 0], "i1")), awkward1.layout.Index32(numpy.array([0, 1, 2, 3, 4], "i4")), [content1, content2])
    assert awkward1.to_list(a) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_arrow(a).to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(awkward1.from_arrow(awkward1.to_arrow(a))) == [1.1, 2.2, 3.3, 4.4, 5.5]

    a = pyarrow.UnionArray.from_sparse(pyarrow.array([0, 0, 0, 0, 0], type=pyarrow.int8()), [pyarrow.array([0.0, 1.1, None, 3.3, 4.4]), pyarrow.array([True, None, False, True, False])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [0.0, 1.1, None, 3.3, 4.4]

    uniontype = pyarrow.union([pyarrow.field("0", pyarrow.list_(pyarrow.float64())),
                               pyarrow.field("1", pyarrow.float64())],
                              "sparse",
                              [0, 1])
    a = pyarrow.Array.from_buffers(
            uniontype,
            5,
            [pyarrow.py_buffer(numpy.array([3], "u1")),
             pyarrow.py_buffer(numpy.array([0, 1, 0, 1, 1], "i1")),
             None],
            children=[pyarrow.array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]),
                      pyarrow.array([0.0, 1.1, 2.2, 3.3, 4.4])])
    assert a.to_pylist() == [[0.0, 1.1, 2.2], 1.1, None, None, None]
    assert awkward1.to_list(awkward1.from_arrow(a)) == [[0.0, 1.1, 2.2], 1.1, None, None, None]

    a = pyarrow.chunked_array([pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5])])
    assert awkward1.to_list(awkward1.from_arrow(a, highlevel=False)) == [1.1, 2.2, 3.3, 4.4, 5.5]
