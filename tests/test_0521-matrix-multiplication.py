# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_regular():
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

    assert numpy.matmul(a22, b22).tolist() == numpy.matmul(awkward1.Array(a22), awkward1.Array(b22)).tolist()
    assert numpy.matmul(a33, b33).tolist() == numpy.matmul(awkward1.Array(a33), awkward1.Array(b33)).tolist()
    assert numpy.matmul(a22, b23).tolist() == numpy.matmul(awkward1.Array(a22), awkward1.Array(b23)).tolist()
    assert numpy.matmul(a23, b33).tolist() == numpy.matmul(awkward1.Array(a23), awkward1.Array(b33)).tolist()
    assert numpy.matmul(a33, b32).tolist() == numpy.matmul(awkward1.Array(a33), awkward1.Array(b32)).tolist()

    assert numpy.matmul(numpy.array([a22, i22, i22, a22]), numpy.array([b22, i22, b22, i22])).tolist() == numpy.matmul(awkward1.Array(numpy.array([a22, i22, i22, a22])), awkward1.Array(numpy.array([b22, i22, b22, i22]))).tolist()


numba = pytest.importorskip("numba")

def test_irregular():
    i22 = numpy.array([[1, 0], [0, 1]])
    a22 = numpy.array([[1, 2], [3, 4]])
    b22 = numpy.array([[5, 6], [7, 8]])

    assert numpy.matmul(numpy.array([a22, i22, i22, a22]), numpy.array([b22, i22, b22, i22])).tolist() == numpy.matmul(awkward1.Array([a22, i22, i22, a22]), awkward1.Array([b22, i22, b22, i22])).tolist()

    lefts = awkward1.Array([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[1], [2], [3], [4]],
    ])
    rights = awkward1.Array([
        [[7, 8, 9], [10, 11, 12]],
        [[8, 10], [11, 12], [13, 14], [15, 16]],
        [[5, 6, 7]],
    ])

    assert numpy.matmul(lefts, rights).tolist() == [
        [[ 27,  30,  33],
         [ 61,  68,  75],
         [ 95, 106, 117]],
        [[129, 140],
         [317, 348]],
        [[ 5,  6,  7],
         [10, 12, 14],
         [15, 18, 21],
         [20, 24, 28]]
    ]
