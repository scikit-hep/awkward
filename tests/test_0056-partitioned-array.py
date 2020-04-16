# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_basic():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert len(array) == 8
    assert [awkward1.to_list(x) for x in array.partitions] == [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [[6.6], [], [], [], [7.7, 8.8, 9.9]]]
    assert awkward1.to_list(array.partition(0)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(array.partition(1)) == [[6.6], [], [], [], [7.7, 8.8, 9.9]]
    assert array.start(0) == 0
    assert array.start(1) == 3
    assert array.stop(0) == 3
    assert array.stop(1) == 8
    assert array.partitionid_index_at(0) == (0, 0)
    assert array.partitionid_index_at(1) == (0, 1)
    assert array.partitionid_index_at(2) == (0, 2)
    assert array.partitionid_index_at(3) == (1, 0)
    assert array.partitionid_index_at(4) == (1, 1)
    assert array.partitionid_index_at(5) == (1, 2)
    assert array.partitionid_index_at(6) == (1, 3)
    assert array.partitionid_index_at(7) == (1, 4)

    assert array.tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"

    assert awkward1.to_list(array._ext.getitem_at(0))  == [1.1, 2.2, 3.3]
    assert awkward1.to_list(array._ext.getitem_at(1))  == []
    assert awkward1.to_list(array._ext.getitem_at(2))  == [4.4, 5.5]
    assert awkward1.to_list(array._ext.getitem_at(3))  == [6.6]
    assert awkward1.to_list(array._ext.getitem_at(4))  == []
    assert awkward1.to_list(array._ext.getitem_at(5))  == []
    assert awkward1.to_list(array._ext.getitem_at(6))  == []
    assert awkward1.to_list(array._ext.getitem_at(7))  == [7.7, 8.8, 9.9]
    assert awkward1.to_list(array._ext.getitem_at(-1)) == [7.7, 8.8, 9.9]
    assert awkward1.to_list(array._ext.getitem_at(-2)) == []
    assert awkward1.to_list(array._ext.getitem_at(-3)) == []
    assert awkward1.to_list(array._ext.getitem_at(-4)) == []
    assert awkward1.to_list(array._ext.getitem_at(-5)) == [6.6]
    assert awkward1.to_list(array._ext.getitem_at(-6)) == [4.4, 5.5]
    assert awkward1.to_list(array._ext.getitem_at(-7)) == []
    assert awkward1.to_list(array._ext.getitem_at(-8)) == [1.1, 2.2, 3.3]

    assert array._ext.getitem_range(0, 8, 1).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(1, 8, 1).tojson()  == "[[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(2, 8, 1).tojson()  == "[[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(3, 8, 1).tojson()  == "[[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(4, 8, 1).tojson()  == "[[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(5, 8, 1).tojson()  == "[[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(6, 8, 1).tojson()  == "[[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(7, 8, 1).tojson()  == "[[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(8, 8, 1).tojson()  == "[]"
    assert array._ext.getitem_range(-1, 8, 1).tojson() == "[[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-2, 8, 1).tojson() == "[[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-3, 8, 1).tojson() == "[[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-4, 8, 1).tojson() == "[[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-5, 8, 1).tojson() == "[[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-6, 8, 1).tojson() == "[[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-7, 8, 1).tojson() == "[[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-8, 8, 1).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"

    assert array._ext.getitem_range(0, 8, 1).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(0, 7, 1).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[]]"
    assert array._ext.getitem_range(0, 6, 1).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[]]"
    assert array._ext.getitem_range(0, 5, 1).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[]]"
    assert array._ext.getitem_range(0, 4, 1).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6]]"
    assert array._ext.getitem_range(0, 3, 1).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5]]"
    assert array._ext.getitem_range(0, 2, 1).tojson()  == "[[1.1,2.2,3.3],[]]"
    assert array._ext.getitem_range(0, 1, 1).tojson()  == "[[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(0, 0, 1).tojson()  == "[]"
    assert array._ext.getitem_range(0, -8, 1).tojson() == "[]"
    assert array._ext.getitem_range(0, -7, 1).tojson() == "[[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(0, -6, 1).tojson() == "[[1.1,2.2,3.3],[]]"
    assert array._ext.getitem_range(0, -5, 1).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5]]"
    assert array._ext.getitem_range(0, -4, 1).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6]]"
    assert array._ext.getitem_range(0, -3, 1).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[]]"
    assert array._ext.getitem_range(0, -2, 1).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[]]"
    assert array._ext.getitem_range(0, -1, 1).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[]]"

    assert array._ext.getitem_range(0, 8, 2).tojson()  == "[[1.1,2.2,3.3],[4.4,5.5],[],[]]"
    assert array._ext.getitem_range(1, 8, 2).tojson()  == "[[],[6.6],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(1, 7, 2).tojson()  == "[[],[6.6],[]]"
    assert array._ext.getitem_range(2, 8, 2).tojson()  == "[[4.4,5.5],[],[]]"
    assert array._ext.getitem_range(3, 8, 2).tojson()  == "[[6.6],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(0, 8, 3).tojson()  == "[[1.1,2.2,3.3],[6.6],[]]"
    assert array._ext.getitem_range(1, 8, 3).tojson()  == "[[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(1, 7, 3).tojson()  == "[[],[]]"
    assert array._ext.getitem_range(2, 8, 3).tojson()  == "[[4.4,5.5],[]]"
    assert array._ext.getitem_range(3, 8, 3).tojson()  == "[[6.6],[]]"

    assert array._ext.getitem_range(2, 0, -1).tojson()  == "[[4.4,5.5],[]]"
    assert array._ext.getitem_range(2, None, -1).tojson()  == "[[4.4,5.5],[],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(1, None, -1).tojson()  == "[[],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(3, None, -1).tojson()  == "[[6.6],[4.4,5.5],[],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(None, None, -1).tojson()  == "[[7.7,8.8,9.9],[],[],[],[6.6],[4.4,5.5],[],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(-1, None, -1).tojson()  == "[[7.7,8.8,9.9],[],[],[],[6.6],[4.4,5.5],[],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(-2, None, -1).tojson()  == "[[],[],[],[6.6],[4.4,5.5],[],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(-2, 0, -1).tojson()  == "[[],[],[],[6.6],[4.4,5.5],[]]"
    assert array._ext.getitem_range(-2, 1, -1).tojson()  == "[[],[],[],[6.6],[4.4,5.5]]"
    assert array._ext.getitem_range(-2, 2, -1).tojson()  == "[[],[],[],[6.6]]"
    assert array._ext.getitem_range(-1, 3, -1).tojson()  == "[[7.7,8.8,9.9],[],[],[]]"
    assert array._ext.getitem_range(-1, None, -2).tojson()  == "[[7.7,8.8,9.9],[],[6.6],[]]"
    assert array._ext.getitem_range(-2, None, -2).tojson()  == "[[],[],[4.4,5.5],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(-1, None, -3).tojson()  == "[[7.7,8.8,9.9],[],[]]"
    assert array._ext.getitem_range(-2, None, -3).tojson()  == "[[],[6.6],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(-3, None, -3).tojson()  == "[[],[4.4,5.5]]"
    assert array._ext.getitem_range(-4, None, -3).tojson()  == "[[],[]]"
    assert array._ext.getitem_range(-5, None, -3).tojson()  == "[[6.6],[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(-2, 0, -2).tojson()  == "[[],[],[4.4,5.5]]"

    assert [awkward1.to_list(x) for x in array] == [[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]

    assert awkward1.to_list(array.toContent()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [], [], [], [7.7, 8.8, 9.9]]

def test_range_slices():
    a1 = awkward1.from_iter(numpy.array([0, 1, 2], dtype=numpy.int64), highlevel=False)
    a2 = awkward1.from_iter(numpy.array([3, 4], dtype=numpy.int64), highlevel=False)
    a3 = awkward1.from_iter(numpy.array([5], dtype=numpy.int64), highlevel=False)
    a4 = awkward1.from_iter(numpy.array([], dtype=numpy.int64), highlevel=False)
    a5 = awkward1.from_iter(numpy.array([6, 7, 8, 9], dtype=numpy.int64), highlevel=False)
    aspart = awkward1.partition.IrregularlyPartitionedArray([a1, a2, a3, a4, a5])
    asfull = awkward1.concatenate([a1, a2, a3, a4, a5], highlevel=False)
    aslist = awkward1.to_list(asfull)

    for start in range(10):
        for stop in range(10):
            for step in (1, 2, 3, 4, 5, -1, -2, -3, -4, -5):
                assert awkward1.to_list(asfull[start:stop:step]) == aslist[start:stop:step]
                assert aspart._ext.getitem_range(start, stop, step).tojson() == asfull[start:stop:step].tojson()

def test_as_slice():
    one = awkward1.from_iter([False, True, False], highlevel=False)
    two = awkward1.from_iter([True, True, True, False, False], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    target = awkward1.Array([0, 1, 2, 100, 200, 300, 400, 500])
    assert awkward1.to_list(target[array]) == [1, 100, 200, 300]

def test_repartition():
    one = awkward1.from_iter([0, 1, 2], highlevel=False)
    two = awkward1.from_iter([100, 200, 300, 400, 500], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert [list(x) for x in array.repartition([3, 8]).partitions] == [[0, 1, 2], [100, 200, 300, 400, 500]]
    assert [list(x) for x in array.repartition([8]).partitions] == [[0, 1, 2, 100, 200, 300, 400, 500]]
    assert [list(x) for x in array.repartition([4, 5, 8]).partitions] == [[0, 1, 2, 100], [200], [300, 400, 500]]
    assert [list(x) for x in array.repartition([4, 5, 5, 8]).partitions] == [[0, 1, 2, 100], [200], [], [300, 400, 500]]
    assert [list(x) for x in array.repartition([2, 8]).partitions] == [[0, 1], [2, 100, 200, 300, 400, 500]]
    assert [list(x) for x in array.repartition([2, 5, 8]).partitions] == [[0, 1], [2, 100, 200], [300, 400, 500]]

def test_getitem_basic():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert awkward1.to_list(array[0]) == [1.1, 2.2, 3.3]
    assert awkward1.to_list(array[1]) == []
    assert awkward1.to_list(array[2]) == [4.4, 5.5]
    assert awkward1.to_list(array[3]) == [6.6]
    assert awkward1.to_list(array[4]) == []
    assert awkward1.to_list(array[5]) == []
    assert awkward1.to_list(array[6]) == []
    assert awkward1.to_list(array[7]) == [7.7, 8.8, 9.9]
    assert awkward1.to_list(array[-1]) == [7.7, 8.8, 9.9]
    assert awkward1.to_list(array[-2]) == []
    assert awkward1.to_list(array[-3]) == []
    assert awkward1.to_list(array[-4]) == []
    assert awkward1.to_list(array[-5]) == [6.6]
    assert awkward1.to_list(array[-6]) == [4.4, 5.5]
    assert awkward1.to_list(array[-7]) == []
    assert awkward1.to_list(array[-8]) == [1.1, 2.2, 3.3]

    assert array[:].tojson()   == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array[2:6].tojson() == "[[4.4,5.5],[6.6],[],[]]"

    assert array[1:].tojson()  == "[[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array[2:].tojson()  == "[[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array[3:].tojson()  == "[[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array[4:].tojson()  == "[[],[],[],[7.7,8.8,9.9]]"
    assert array[5:].tojson()  == "[[],[],[7.7,8.8,9.9]]"
    assert array[6:].tojson()  == "[[],[7.7,8.8,9.9]]"
    assert array[7:].tojson()  == "[[7.7,8.8,9.9]]"
    assert array[8:].tojson()  == "[]"
    assert array[-1:].tojson() == "[[7.7,8.8,9.9]]"
    assert array[-2:].tojson() == "[[],[7.7,8.8,9.9]]"
    assert array[-3:].tojson() == "[[],[],[7.7,8.8,9.9]]"
    assert array[-4:].tojson() == "[[],[],[],[7.7,8.8,9.9]]"
    assert array[-5:].tojson() == "[[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array[-6:].tojson() == "[[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array[-7:].tojson() == "[[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array[-8:].tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"

    assert array[:-1].tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[]]"
    assert array[:-2].tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[]]"
    assert array[:-3].tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[]]"
    assert array[:-4].tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6]]"
    assert array[:-5].tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5]]"
    assert array[:-6].tojson() == "[[1.1,2.2,3.3],[]]"
    assert array[:-7].tojson() == "[[1.1,2.2,3.3]]"
    assert array[:-8].tojson() == "[]"
    assert array[:0].tojson()  == "[]"
    assert array[:1].tojson()  == "[[1.1,2.2,3.3]]"
    assert array[:2].tojson()  == "[[1.1,2.2,3.3],[]]"
    assert array[:3].tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5]]"
    assert array[:4].tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6]]"
    assert array[:5].tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[]]"
    assert array[:6].tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[]]"
    assert array[:7].tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[]]"
    assert array[:8].tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"

    one = awkward1.from_iter([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], highlevel=False)
    two = awkward1.from_iter([{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    if not awkward1._util.py27 and not awkward1._util.py35:
        assert array.tojson() == '[{"x":0.0,"y":[]},{"x":1.1,"y":[1]},{"x":2.2,"y":[2,2]},{"x":3.3,"y":[3,3,3]},{"x":4.4,"y":[4,4,4,4]}]'
        assert array["x"].tojson() == "[0.0,1.1,2.2,3.3,4.4]"
        assert array["y"].tojson() == "[[],[1],[2,2],[3,3,3],[4,4,4,4]]"
        assert array[["x"]].tojson() == '[{"x":0.0},{"x":1.1},{"x":2.2},{"x":3.3},{"x":4.4}]'

def test_getitem_first_dimension_int():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert awkward1.to_list(array[0,]) == [1.1, 2.2, 3.3]
    assert awkward1.to_list(array[-8,]) == [1.1, 2.2, 3.3]
    assert array[0, 1] == 2.2
    assert array[-8, 1] == 2.2
    assert awkward1.to_list(array[0, [-1, 0]]) == [3.3, 1.1]
    assert awkward1.to_list(array[0, [False, True, True]]) == [2.2, 3.3]
    assert awkward1.to_list(array[7,]) == [7.7, 8.8, 9.9]
    assert awkward1.to_list(array[-1,]) == [7.7, 8.8, 9.9]
    assert array[7, 1] == 8.8
    assert array[-1, 1] == 8.8
    assert awkward1.to_list(array[7, [-1, 0]]) == [9.9, 7.7]
    assert awkward1.to_list(array[7, [False, True, True]]) == [8.8, 9.9]

def test_getitem_first_dimension_slice():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert array[2:6,].tojson() == "[[4.4,5.5],[6.6],[],[]]"
    assert array[::-1,].tojson() == "[[7.7,8.8,9.9],[],[],[],[6.6],[4.4,5.5],[],[1.1,2.2,3.3]]"
    assert array[::-1, :2].tojson() == "[[7.7,8.8],[],[],[],[6.6],[4.4,5.5],[],[1.1,2.2]]"
    assert array[::-1, :1].tojson() == "[[7.7],[],[],[],[6.6],[4.4],[],[1.1]]"

def test_getitem_first_dimension_field():
    one = awkward1.from_iter([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], highlevel=False)
    two = awkward1.from_iter([{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    if not awkward1._util.py27 and not awkward1._util.py35:
        assert array.tojson() == '[{"x":0.0,"y":[]},{"x":1.1,"y":[1]},{"x":2.2,"y":[2,2]},{"x":3.3,"y":[3,3,3]},{"x":4.4,"y":[4,4,4,4]}]'
        assert array["y", :, :2].tojson() == "[[],[1],[2,2],[3,3],[4,4]]"
        assert array[["y"], :, :2].tojson() == '[{"y":[]},{"y":[1]},{"y":[2,2]},{"y":[3,3]},{"y":[4,4]}]'
        assert array[:, "y", :2].tojson() == "[[],[1],[2,2],[3,3],[4,4]]"
        assert array["y", ..., :2].tojson() == "[[],[1],[2,2],[3,3],[4,4]]"
        assert array[numpy.newaxis, "y", :, :2].tojson() == "[[[],[1],[2,2],[3,3],[4,4]]]"

def test_getitem_first_dimension_intarray():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert isinstance(array[[-1, 3, 2, 0, 2, 3, 7]], awkward1.layout.ListArray64)
    assert awkward1.to_list(array[[-1, 3, 2, 0, 2, 3, 7]]) == [[7.7, 8.8, 9.9], [6.6], [4.4, 5.5], [1.1, 2.2, 3.3], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

    m1a = awkward1.from_iter([-1, 3, 2], highlevel=False)
    m2a = awkward1.from_iter([0, 2, 3, 7], highlevel=False)
    ma = awkward1.partition.IrregularlyPartitionedArray([m1a, m2a])
    assert isinstance(array[ma], awkward1.layout.ListArray64)
    assert awkward1.to_list(array[ma]) == [[7.7, 8.8, 9.9], [6.6], [4.4, 5.5], [1.1, 2.2, 3.3], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

    m1b = awkward1.from_iter([-1, 3, 2, 0, 2], highlevel=False)
    m2b = awkward1.from_iter([3, 7], highlevel=False)
    mb = awkward1.partition.IrregularlyPartitionedArray([m1b, m2b])
    assert isinstance(array[mb], awkward1.layout.ListArray64)
    assert awkward1.to_list(array[mb]) == [[7.7, 8.8, 9.9], [6.6], [4.4, 5.5], [1.1, 2.2, 3.3], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

    assert isinstance(array[[-1, 3, None, 0, 2, 3, 7]], awkward1.layout.IndexedOptionArray64)
    assert awkward1.to_list(array[[-1, 3, None, 0, 2, 3, 7]]) == [[7.7, 8.8, 9.9], [6.6], None, [1.1, 2.2, 3.3], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]

    m1a = awkward1.from_iter([-1, 3, 2], highlevel=False)
    m2a = awkward1.from_iter([0, None, 3, 7], highlevel=False)
    ma = awkward1.partition.IrregularlyPartitionedArray([m1a, m2a])
    assert isinstance(array[ma], awkward1.layout.IndexedOptionArray64)
    assert awkward1.to_list(array[ma]) == [[7.7, 8.8, 9.9], [6.6], [4.4, 5.5], [1.1, 2.2, 3.3], None, [6.6], [7.7, 8.8, 9.9]]

    m1b = awkward1.from_iter([-1, 3, 2, 0, None], highlevel=False)
    m2b = awkward1.from_iter([3, 7], highlevel=False)
    mb = awkward1.partition.IrregularlyPartitionedArray([m1b, m2b])
    assert isinstance(array[mb], awkward1.layout.IndexedOptionArray64)
    assert awkward1.to_list(array[mb]) == [[7.7, 8.8, 9.9], [6.6], [4.4, 5.5], [1.1, 2.2, 3.3], None, [6.6], [7.7, 8.8, 9.9]]

def test_getitem_first_dimension_boolarray():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert isinstance(array[[True, False, True, True, False, True, False, True]], awkward1.partition.IrregularlyPartitionedArray)
    assert array[[True, False, True, True, False, True, False, True]].tojson() == "[[1.1,2.2,3.3],[4.4,5.5],[6.6],[],[7.7,8.8,9.9]]"

    m1a = awkward1.from_iter([True, False, True], highlevel=False)
    m2a = awkward1.from_iter([True, False, True, False, True], highlevel=False)
    ma = awkward1.partition.IrregularlyPartitionedArray([m1a, m2a])
    assert isinstance(array[ma], awkward1.partition.IrregularlyPartitionedArray)
    assert array[ma].tojson() == "[[1.1,2.2,3.3],[4.4,5.5],[6.6],[],[7.7,8.8,9.9]]"

    m1b = awkward1.from_iter([True, False, True, True, False], highlevel=False)
    m2b = awkward1.from_iter([True, False, True], highlevel=False)
    mb = awkward1.partition.IrregularlyPartitionedArray([m1b, m2b])
    assert isinstance(array[mb], awkward1.partition.IrregularlyPartitionedArray)
    assert array[mb].tojson() == "[[1.1,2.2,3.3],[4.4,5.5],[6.6],[],[7.7,8.8,9.9]]"

def test_getitem_first_dimension_jaggedarray():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert isinstance(array[[[2, 0], [], [1], [0, 0], [], [], [], [2, 1, 1, 2]]], awkward1.partition.IrregularlyPartitionedArray)
    assert array[[[2, 0], [], [1], [0, 0], [], [], [], [2, 1, 1, 2]]].tojson() == "[[3.3,1.1],[],[5.5],[6.6,6.6],[],[],[],[9.9,8.8,8.8,9.9]]"

    m1a = awkward1.from_iter([[2, 0], [], [1]], highlevel=False)
    m2a = awkward1.from_iter([[0, 0], [], [], [], [2, 1, 1, 2]], highlevel=False)
    ma = awkward1.partition.IrregularlyPartitionedArray([m1a, m2a])
    assert isinstance(array[ma], awkward1.partition.IrregularlyPartitionedArray)
    assert array[ma].tojson() == "[[3.3,1.1],[],[5.5],[6.6,6.6],[],[],[],[9.9,8.8,8.8,9.9]]"

    m1b = awkward1.from_iter([[2, 0], [], [1], [0, 0], []], highlevel=False)
    m2b = awkward1.from_iter([[], [], [2, 1, 1, 2]], highlevel=False)
    mb = awkward1.partition.IrregularlyPartitionedArray([m1b, m2b])
    assert isinstance(array[mb], awkward1.partition.IrregularlyPartitionedArray)
    assert array[mb].tojson() == "[[3.3,1.1],[],[5.5],[6.6,6.6],[],[],[],[9.9,8.8,8.8,9.9]]"

def test_highlevel():
    one = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = awkward1.from_iter([[6.6], [], [], [], [7.7, 8.8, 9.9]], highlevel=False)
    layout = awkward1.partition.IrregularlyPartitionedArray([one, two])
    array = awkward1.Array(layout)

    assert awkward1.to_list(array) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [], [], [], [7.7, 8.8, 9.9]]

def test_mask():
    array = awkward1.Array([1, 2, 3, 4, 5])
    mask = awkward1.Array([False, True, True, True, False])
    assert awkward1.to_list(awkward1.mask(array, mask)) == [None, 2, 3, 4, None]

def test_indexed():
    content = awkward1.from_iter([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], highlevel=False)
    index = awkward1.layout.Index64(numpy.array([3, 2, 2, 4, 9, 8, 0, 7, 7, 5, 2, 2, 6, 7, 3], dtype=numpy.int64))
    indexedarray = awkward1.layout.IndexedArray64(index, content)
    array = awkward1.partition.IrregularlyPartitionedArray([indexedarray])
    assert awkward1.to_list(array) == [3.3, 2.2, 2.2, 4.4, 9.9, 8.8, 0.0, 7.7, 7.7, 5.5, 2.2, 2.2, 6.6, 7.7, 3.3]

    array2 = array.repartition([5, 6, 8, 10, 10, 15])
    assert array2.numpartitions == 6
    assert awkward1.to_list(array2) == [3.3, 2.2, 2.2, 4.4, 9.9, 8.8, 0.0, 7.7, 7.7, 5.5, 2.2, 2.2, 6.6, 7.7, 3.3]
