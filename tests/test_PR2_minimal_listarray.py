# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import numpy

import awkward1

def test():
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], dtype="i4"))
    assert numpy.asarray(offsets).tolist() == [0, 2, 2, 3]
#     assert offsets[0] == 0
#     assert offsets[1] == 2
#     assert offsets[2] == 2
#     assert offsets[3] == 3
#
#     content = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))
#     assert numpy.asarray(content).tolist() == [[0,  1,  2,  3],
#                                                [4,  5,  6,  7],
#                                                [8,  9, 10, 11]]
#     assert numpy.asarray(content[0]).tolist() == [0,  1,  2,  3]
#     assert numpy.asarray(content[1]).tolist() == [4,  5,  6,  7]
#     assert numpy.asarray(content[2]).tolist() == [8,  9, 10, 11]
#     assert [content[i][j] for i in range(3) for j in range(4)] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#
#     array = awkward1.layout.ListOffsetArray(offsets, content)
#     assert numpy.asarray(array[0]).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
#     assert numpy.asarray(array[1]).tolist() == []
#     assert numpy.asarray(array[2]).tolist() == [[8, 9, 10, 11]]
#     assert numpy.asarray(array[1:3][0]).tolist() == []
#     assert numpy.asarray(array[1:3][1]).tolist() == [[8, 9, 10, 11]]
#     assert numpy.asarray(array[2:3][0]).tolist() == [[8, 9, 10, 11]]
#
# def test_len():
#     offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], dtype="i4"))
#     content = awkward1.layout.NumpyArray(numpy.arange(12).reshape(4, 3))
#     array = awkward1.layout.ListOffsetArray(offsets, content)
#     assert len(content) == 4
#     assert len(array) == 3
#
# def test_members():
#     offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], dtype="i4"))
#     content = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))
#     array = awkward1.layout.ListOffsetArray(offsets, content)
#     assert numpy.asarray(array.offsets).tolist() == [0, 2, 2, 3]
#     assert numpy.asarray(array.content).tolist() == [[0,  1,  2,  3], [4,  5,  6,  7], [8,  9, 10, 11]]
#     array2 = awkward1.layout.ListOffsetArray(offsets, array)
#     assert numpy.asarray(array2.offsets).tolist() == [0, 2, 2, 3]
#     assert numpy.asarray(array2.content.offsets).tolist() == [0, 2, 2, 3]
#     assert numpy.asarray(array2.content.content).tolist() == [[0,  1,  2,  3], [4,  5,  6,  7], [8,  9, 10, 11]]
