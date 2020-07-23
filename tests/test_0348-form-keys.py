# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
try:
    # pybind11 only supports cPickle protocol 2+ (-1 in pickle.dumps)
    # (automatically satisfied in Python 3; this is just to keep testing Python 2.7)
    import cPickle as pickle
except ImportError:
    import pickle

import pytest
import numpy

import awkward1


def test_numpyarray():
    assert awkward1.from_arrayset(*awkward1.to_arrayset([1, 2, 3, 4, 5])).tolist() == [1, 2, 3, 4, 5]
    assert pickle.loads(pickle.dumps(awkward1.Array([1, 2, 3, 4, 5]), -1)).tolist() == [1, 2, 3, 4, 5]


def test_listoffsetarray():
    assert awkward1.from_arrayset(*awkward1.to_arrayset([[1, 2, 3], [], [4, 5]])).tolist() == [[1, 2, 3], [], [4, 5]]
    assert awkward1.from_arrayset(*awkward1.to_arrayset(["one", "two", "three", "four", "five"])).tolist() == ["one", "two", "three", "four", "five"]
    assert awkward1.from_arrayset(*awkward1.to_arrayset([["one", "two", "three"], [], ["four", "five"]])).tolist() == [["one", "two", "three"], [], ["four", "five"]]
    assert pickle.loads(pickle.dumps(awkward1.Array([[1, 2, 3], [], [4, 5]]), -1)).tolist() == [[1, 2, 3], [], [4, 5]]


def test_listarray():
    listoffsetarray = awkward1.Array([[1, 2, 3], [], [4, 5]]).layout
    listarray = awkward1.layout.ListArray64(listoffsetarray.starts, listoffsetarray.stops, listoffsetarray.content)
    assert awkward1.from_arrayset(*awkward1.to_arrayset(listarray)).tolist() == [[1, 2, 3], [], [4, 5]]
    assert pickle.loads(pickle.dumps(awkward1.Array(listarray), -1)).tolist() == [[1, 2, 3], [], [4, 5]]


def test_indexedoptionarray():
    assert awkward1.from_arrayset(*awkward1.to_arrayset([1, 2, 3, None, None, 5])).tolist() == [1, 2, 3, None, None, 5]
    assert pickle.loads(pickle.dumps(awkward1.Array([1, 2, 3, None, None, 5]), -1)).tolist() == [1, 2, 3, None, None, 5]


def test_indexedarray():
    content = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    index = awkward1.layout.Index64(numpy.array([3, 1, 1, 4, 2], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    assert awkward1.from_arrayset(*awkward1.to_arrayset(indexedarray)).tolist() == [3.3, 1.1, 1.1, 4.4, 2.2]
    assert pickle.loads(pickle.dumps(awkward1.Array(indexedarray), -1)).tolist() == [3.3, 1.1, 1.1, 4.4, 2.2]


def test_emptyarray():
    assert awkward1.from_arrayset(*awkward1.to_arrayset([])).tolist() == []
    assert awkward1.from_arrayset(*awkward1.to_arrayset([[], [], []])).tolist() == [[], [], []]

    assert pickle.loads(pickle.dumps(awkward1.Array([]), -1)).tolist() == []
    assert pickle.loads(pickle.dumps(awkward1.Array([[], [], []]), -1)).tolist() == [[], [], []]


def test_bytemaskedarray():
    content = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    mask = awkward1.layout.Index8(numpy.array([False, True, True, False, False], dtype=numpy.int8))
    bytemaskedarray = awkward1.layout.ByteMaskedArray(mask, content, True)
    assert awkward1.from_arrayset(*awkward1.to_arrayset(bytemaskedarray)).tolist() == [None, 1.1, 2.2, None, None]
    assert pickle.loads(pickle.dumps(awkward1.Array(bytemaskedarray), -1)).tolist() == [None, 1.1, 2.2, None, None]


def test_bitmaskedarray():
    content = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    mask = awkward1.layout.IndexU8(numpy.packbits(numpy.array([False, True, True, False, False], dtype=numpy.int8)))
    bitmaskedarray = awkward1.layout.BitMaskedArray(mask, content, True, 5, False)
    assert awkward1.from_arrayset(*awkward1.to_arrayset(bitmaskedarray)).tolist() == [None, 1.1, 2.2, None, None]
    assert pickle.loads(pickle.dumps(awkward1.Array(bitmaskedarray), -1)).tolist() == [None, 1.1, 2.2, None, None]

def test_recordarray():
    assert awkward1.from_arrayset(*awkward1.to_arrayset([(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])])) == [(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]
    assert awkward1.from_arrayset(*awkward1.to_arrayset([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}])) == [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]

    assert pickle.loads(pickle.dumps(awkward1.Array([(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]), -1)) == [(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]
    assert pickle.loads(pickle.dumps(awkward1.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]), -1)) == [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]


def test_record():
    assert pickle.loads(pickle.dumps(awkward1.Record({"x": 2.2, "y": [1, 2]}), -1)) == {"x": 2.2, "y": [1, 2]}
    assert pickle.loads(pickle.dumps(awkward1.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}])[1], -1)) == {"x": 2.2, "y": [1, 2]}


def test_regulararray():
    content = awkward1.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).layout
    regulararray = awkward1.layout.RegularArray(content, 3)
    assert awkward1.from_arrayset(*awkward1.to_arrayset(regulararray)).tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    assert pickle.loads(pickle.dumps(awkward1.Array(regulararray), -1)).tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]


def test_unionarray():
    assert awkward1.from_arrayset(*awkward1.to_arrayset([[1, 2, 3], [], 4, 5])).tolist() == [[1, 2, 3], [], 4, 5]
    assert pickle.loads(pickle.dumps(awkward1.Array([[1, 2, 3], [], 4, 5]), -1)).tolist() == [[1, 2, 3], [], 4, 5]


def test_unmaskedarray():
    content = awkward1.Array([1, 2, 3, 4, 5]).layout
    unmaskedarray = awkward1.layout.UnmaskedArray(content)
    assert awkward1.from_arrayset(*awkward1.to_arrayset(unmaskedarray)).tolist() == [1, 2, 3, 4, 5]
    assert pickle.loads(pickle.dumps(awkward1.Array(unmaskedarray), -1)).tolist() == [1, 2, 3, 4, 5]


def test_partitioned():
    array = awkward1.repartition(awkward1.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3)

    form, container, num_partitions = awkward1.to_arrayset(array, partition_first=True)
    assert awkward1.from_arrayset(form, container, num_partitions, partition_first=True).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    form, container, num_partitions = awkward1.to_arrayset(array, partition_first=False)
    assert awkward1.from_arrayset(form, container, num_partitions, partition_first=False).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    one = awkward1.Array([1, 2, 3, 4, 5])
    two = awkward1.Array([6, 7, 8, 9, 10])
    container = {}
    form1, _, _ = awkward1.to_arrayset(one, container, 0)
    form2, _, _ = awkward1.to_arrayset(two, container, 1)
    assert form1 == form2

    assert awkward1.from_arrayset(form1, container, 2).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert pickle.loads(pickle.dumps(array, -1)).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_lazy():
    array = awkward1.Array([1, 2, 3, 4, 5])

    form, container, num_partitions = awkward1.to_arrayset(array)

    assert awkward1.from_arrayset(form, container, num_partitions, lazy=True, lazy_lengths=5).tolist() == [1, 2, 3, 4, 5]


def test_lazy_partitioned():
    array = awkward1.repartition(awkward1.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3)
    form, container, num_partitions = awkward1.to_arrayset(array)
    assert num_partitions == 4

    assert awkward1.from_arrayset(form, container, num_partitions, lazy=True, lazy_lengths=[3, 3, 3, 1]).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
