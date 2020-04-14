# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test():
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

    assert array._ext.getitem_range(0, 8).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(1, 8).tojson()  == "[[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(2, 8).tojson()  == "[[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(3, 8).tojson()  == "[[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(4, 8).tojson()  == "[[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(5, 8).tojson()  == "[[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(6, 8).tojson()  == "[[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(7, 8).tojson()  == "[[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(8, 8).tojson()  == "[]"
    assert array._ext.getitem_range(-1, 8).tojson() == "[[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-2, 8).tojson() == "[[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-3, 8).tojson() == "[[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-4, 8).tojson() == "[[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-5, 8).tojson() == "[[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-6, 8).tojson() == "[[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-7, 8).tojson() == "[[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(-8, 8).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"

    assert array._ext.getitem_range(0, 8).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]"
    assert array._ext.getitem_range(0, 7).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[]]"
    assert array._ext.getitem_range(0, 6).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[]]"
    assert array._ext.getitem_range(0, 5).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[]]"
    assert array._ext.getitem_range(0, 4).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6]]"
    assert array._ext.getitem_range(0, 3).tojson()  == "[[1.1,2.2,3.3],[],[4.4,5.5]]"
    assert array._ext.getitem_range(0, 2).tojson()  == "[[1.1,2.2,3.3],[]]"
    assert array._ext.getitem_range(0, 1).tojson()  == "[[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(0, 0).tojson()  == "[]"
    assert array._ext.getitem_range(0, -8).tojson() == "[]"
    assert array._ext.getitem_range(0, -7).tojson() == "[[1.1,2.2,3.3]]"
    assert array._ext.getitem_range(0, -6).tojson() == "[[1.1,2.2,3.3],[]]"
    assert array._ext.getitem_range(0, -5).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5]]"
    assert array._ext.getitem_range(0, -4).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6]]"
    assert array._ext.getitem_range(0, -3).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[]]"
    assert array._ext.getitem_range(0, -2).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[]]"
    assert array._ext.getitem_range(0, -1).tojson() == "[[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[]]"

    assert [awkward1.to_list(x) for x in array] == [[1.1,2.2,3.3],[],[4.4,5.5],[6.6],[],[],[],[7.7,8.8,9.9]]
