# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    i22 = numpy.array([[1, 0], [0, 1]])
    ii22 = numpy.array([[2, 0], [0, 2]])
    iii22 = numpy.array([[3, 0], [0, 3]])
    a22 = numpy.array([[1, 2], [3, 4]])
    b22 = numpy.array([[5, 6], [7, 8]])

    a23 = numpy.array([[1, 2, 3], [4, 5, 6]])
    b23 = numpy.array([[7, 8, 9], [10, 11, 12]])

    a32 = numpy.array([[1, 2], [3, 4], [5, 6]])
    b32 = numpy.array([[7, 8], [9, 10], [11, 12]])

    i33 = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ii33 = numpy.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    iii33 = numpy.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    a33 = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b33 = numpy.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

    assert numpy.matmul(a22, b22).tolist() == numpy.matmul(awkward1.Array(a22), awkward1.Array(b22))
    assert numpy.matmul(a33, b33).tolist() == numpy.matmul(awkward1.Array(a33), awkward1.Array(b33))
    assert numpy.matmul(a22, b23).tolist() == numpy.matmul(awkward1.Array(a22), awkward1.Array(b23))
    assert numpy.matmul(a23, b33).tolist() == numpy.matmul(awkward1.Array(a23), awkward1.Array(b33))
    assert numpy.matmul(a33, b32).tolist() == numpy.matmul(awkward1.Array(a33), awkward1.Array(b32))

    assert numpy.matmul(numpy.array([a22, i22, i22, a22]), numpy.array([b22, i22, b22, i22])) == numpy.matmul(awkward1.Array(numpy.array([a22, i22, i22, a22])), awkward1.Array(numpy.array([b22, i22, b22, i22])))

    assert numpy.matmul(awkward1.Array(numpy.array([2])), awkward1.Array(numpy.array([3])))
