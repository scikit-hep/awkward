# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import numpy

import awkward1
import cupy


def test_from_cupy():
    cupy_array_1d = cupy.arange(10)
    cupy_array_2d = cupy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8]])

    ak_cupy_array_1d = awkward1.from_cupy(cupy_array_1d)
    ak_cupy_array_2d = awkward1.from_cupy(cupy_array_2d)

    for i in range(10):
        assert ak_cupy_array_1d[i] == cupy_array_1d[i]

    for i in range(4):
        for j in range(2):
            assert ak_cupy_array_2d[i][j] == cupy_array_2d[i][j]


def test_from_cupy_tolist():
    cupy_array_1d = cupy.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    ak_cupy_array_1d = awkward1.from_cupy(cupy_array_1d)

    assert awkward1.to_list(ak_cupy_array_1d.layout) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
