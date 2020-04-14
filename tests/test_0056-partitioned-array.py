# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_basic():
    one = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    two = awkward1.Array([[6.6], [], [], [], [7.7, 8.8, 9.9]]).layout
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
    assert array._ext.getitem_range(0, 8, 3).tojson()  == "[[1.1,2.2,3.3],[6.6],[]]"
    assert array._ext.getitem_range(1, 8, 3).tojson()  == "[[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(1, 7, 3).tojson()  == "[[],[]]"
    assert array._ext.getitem_range(2, 8, 3).tojson()  == "[[4.4,5.5],[]]"
    assert array._ext.getitem_range(3, 8, 3).tojson()  == "[[6.6],[]]"

    assert [awkward1.to_list(x) for x in array] == [[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]

    assert awkward1.to_list(array.toContent()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [], [], [], [7.7, 8.8, 9.9]]

def test_as_slice():
    one = awkward1.Array([False, True, False]).layout
    two = awkward1.Array([True, True, True, False, False]).layout
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    target = awkward1.Array([0, 1, 2, 100, 200, 300, 400, 500])
    assert awkward1.to_list(target[array]) == [1, 100, 200, 300]

def test_repartition():
    one = awkward1.Array([0, 1, 2]).layout
    two = awkward1.Array([100, 200, 300, 400, 500]).layout
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert [list(x) for x in array.repartition([3, 8]).partitions] == [[0, 1, 2], [100, 200, 300, 400, 500]]
    assert [list(x) for x in array.repartition([8]).partitions] == [[0, 1, 2, 100, 200, 300, 400, 500]]
    assert [list(x) for x in array.repartition([4, 5, 8]).partitions] == [[0, 1, 2, 100], [200], [300, 400, 500]]
    assert [list(x) for x in array.repartition([4, 5, 5, 8]).partitions] == [[0, 1, 2, 100], [200], [], [300, 400, 500]]
    assert [list(x) for x in array.repartition([2, 8]).partitions] == [[0, 1], [2, 100, 200, 300, 400, 500]]
    assert [list(x) for x in array.repartition([2, 5, 8]).partitions] == [[0, 1], [2, 100, 200], [300, 400, 500]]

def test_getitem_basic():
    one = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    two = awkward1.Array([[6.6], [], [], [], [7.7, 8.8, 9.9]]).layout
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

    one = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}]).layout
    two = awkward1.Array([{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]).layout
    array = awkward1.partition.IrregularlyPartitionedArray([one, two])

    assert array.tojson() == '[{"x":0.0,"y":[]},{"x":1.1,"y":[1]},{"x":2.2,"y":[2,2]},{"x":3.3,"y":[3,3,3]},{"x":4.4,"y":[4,4,4,4]}]'
    assert array["x"].tojson() == "[0.0,1.1,2.2,3.3,4.4]"
    assert array["y"].tojson() == "[[],[1],[2,2],[3,3,3],[4,4,4,4]]"
    assert array[["x"]].tojson() == '[{"x":0.0},{"x":1.1},{"x":2.2},{"x":3.3},{"x":4.4}]'

def test_getitem_first_dimension_int():
    one = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    two = awkward1.Array([[6.6], [], [], [], [7.7, 8.8, 9.9]]).layout
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

# def test_getitem_first_dimension_slice():
#     one = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
#     two = awkward1.Array([[6.6], [], [], [], [7.7, 8.8, 9.9]]).layout
#     array = awkward1.partition.IrregularlyPartitionedArray([one, two])
#
#     print(array[2:6,])
#     raise Exception
