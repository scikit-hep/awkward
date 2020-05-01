# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_EmptyArray():
    array = awkward1.layout.EmptyArray()
    assert awkward1.to_list(array.recurse(0, True, True)) == []

def test_NumpyArray():
    array = awkward1.layout.NumpyArray(numpy.array([3.3, 2.2, 1.1, 5.5, 4.4]))
    assert awkward1.to_list(array.recurse(0, True, True)) == [3.3, 2.2, 1.1, 5.5, 4.4]

    array = awkward1.layout.NumpyArray(numpy.array([[3.3, 2.2, 4.4],
                                                    [1.1, 5.5, 3.3]]))
    assert awkward1.to_list(array.recurse(0, True, True)) == [[3.3, 2.2, 4.4],
                                                               [1.1, 5.5, 3.3]]
    assert awkward1.to_list(array.recurse(1, True, True)) == [[3.3, 2.2, 4.4],
                                                               [1.1, 5.5, 3.3]]


def test_IndexedOffsetArray():
    array = awkward1.Array([[ 2.2,  1.1,  3.3],
                            [],
                            [ 4.4,  5.5 ],
                            [ 5.5 ],
                            [-4.4, -5.5, -6.6]]).layout

    assert awkward1.to_list(array.recurse(0, True, True)) == [[ 2.2,  1.1,  3.3],
                            [],
                            [ 4.4,  5.5 ],
                            [ 5.5 ],
                            [-4.4, -5.5, -6.6]]

    assert awkward1.to_list(array.recurse(1, True, True)) == [[ 2.2,  1.1,  3.3],
                            [],
                            [ 4.4,  5.5 ],
                            [ 5.5 ],
                            [-4.4, -5.5, -6.6]]

def test_3d():
    array = awkward1.layout.NumpyArray(numpy.array([
# axis 2:    0       1       2       3       4         # axis 1:
        [[  1.1,    2.2,    3.3,    4.4,    5.5 ],     # 0
         [  6.6,    7.7,    8.8,    9.9,   10.10],     # 1
         [ 11.11,  12.12,  13.13,  14.14,  15.15]],    # 2
        [[ -1.1,   -2.2,   -3.3,   -4.4,   -5.5],      # 3
         [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],      # 4
         [-11.11, -12.12, -13.13, -14.14, -15.15]]]))  # 5

    assert awkward1.to_list(array.recurse(0, True, True)) == [
        [[  1.1,    2.2,    3.3,    4.4,    5.5 ],     # 0
         [  6.6,    7.7,    8.8,    9.9,   10.10],     # 1
         [ 11.11,  12.12,  13.13,  14.14,  15.15]],    # 2
        [[ -1.1,   -2.2,   -3.3,   -4.4,   -5.5],      # 3
         [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],      # 4
         [-11.11, -12.12, -13.13, -14.14, -15.15]]]    # 5

    assert awkward1.to_list(array.recurse(1, True, True)) == [
        [[  1.1,    2.2,    3.3,    4.4,    5.5 ],     # 0
         [  6.6,    7.7,    8.8,    9.9,   10.10],     # 1
         [ 11.11,  12.12,  13.13,  14.14,  15.15]],    # 2
        [[ -1.1,   -2.2,   -3.3,   -4.4,   -5.5],      # 3
         [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],      # 4
         [-11.11, -12.12, -13.13, -14.14, -15.15]]]    # 5

    assert awkward1.to_list(array.recurse(2, True, True)) == [
        [[  1.1,    2.2,    3.3,    4.4,    5.5 ],     # 0
         [  6.6,    7.7,    8.8,    9.9,   10.10],     # 1
         [ 11.11,  12.12,  13.13,  14.14,  15.15]],    # 2
        [[ -1.1,   -2.2,   -3.3,   -4.4,   -5.5],      # 3
         [ -6.6,   -7.7,   -8.8,   -9.9,  -10.1],      # 4
         [-11.11, -12.12, -13.13, -14.14, -15.15]]]    # 5
