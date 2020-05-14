# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import numpy

import awkward1

def test():
    data = numpy.array([0, 2, 2, 3], dtype="i8")
    offsets = awkward1.layout.Index64(data)
    assert numpy.asarray(offsets).tolist() == [0, 2, 2, 3]
    assert offsets[0] == 0
    assert offsets[1] == 2
    assert offsets[2] == 2
    assert offsets[3] == 3
    data[2] = 999
    assert offsets[2] == 999

    data = numpy.array([0, 2, 2, 3], dtype="i4")
    offsets = awkward1.layout.Index32(data)
    assert numpy.asarray(offsets).tolist() == [0, 2, 2, 3]
    assert offsets[0] == 0
    assert offsets[1] == 2
    assert offsets[2] == 2
    assert offsets[3] == 3
    data[2] = 999
    assert offsets[2] == 999

    content = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))
    assert numpy.asarray(content).tolist() == [[0,  1,  2,  3],
                                               [4,  5,  6,  7],
                                               [8,  9, 10, 11]]
    assert numpy.asarray(content[0]).tolist() == [0,  1,  2,  3]
    assert numpy.asarray(content[1]).tolist() == [4,  5,  6,  7]
    assert numpy.asarray(content[2]).tolist() == [8,  9, 10, 11]
    assert [content[i][j] for i in range(3) for j in range(4)] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    data = numpy.array([0, 2, 2, 3], dtype="i4")
    offsets = awkward1.layout.Index32(data)
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert numpy.asarray(array[0]).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert numpy.asarray(array[1]).tolist() == []
    assert numpy.asarray(array[2]).tolist() == [[8, 9, 10, 11]]
    assert numpy.asarray(array[1:3][0]).tolist() == []
    assert numpy.asarray(array[1:3][1]).tolist() == [[8, 9, 10, 11]]
    assert numpy.asarray(array[2:3][0]).tolist() == [[8, 9, 10, 11]]

def test_len():
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], dtype="i4"))
    content = awkward1.layout.NumpyArray(numpy.arange(12).reshape(4, 3))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert len(content) == 4
    assert len(array) == 3

def test_members():
    offsets = awkward1.layout.Index32(numpy.array([0, 2, 2, 3], dtype="i4"))
    content = awkward1.layout.NumpyArray(numpy.arange(12).reshape(3, 4))
    array = awkward1.layout.ListOffsetArray32(offsets, content)
    assert numpy.asarray(array.offsets).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(array.content).tolist() == [[0,  1,  2,  3], [4,  5,  6,  7], [8,  9, 10, 11]]
    array2 = awkward1.layout.ListOffsetArray32(offsets, array)
    assert numpy.asarray(array2.offsets).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(array2.content.offsets).tolist() == [0, 2, 2, 3]
    assert numpy.asarray(array2.content.content).tolist() == [[0,  1,  2,  3], [4,  5,  6,  7], [8,  9, 10, 11]]
