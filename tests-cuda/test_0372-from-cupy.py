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


def test_NumpyArray_constructor():
    assert awkward1.kernels(awkward1.layout.NumpyArray(numpy.array([1, 2, 3]))) == "cpu"
    assert awkward1.kernels(awkward1.layout.NumpyArray(cupy.array([1, 2, 3]))) == "cuda"


def test_add():
    one = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], lib="cuda")
    two = awkward1.Array([100, 200, 300], lib="cuda")
    assert awkward1.kernels(one) == "cuda"
    assert awkward1.kernels(two) == "cuda"
    three = one + two
    assert awkward1.to_list(three) == [[101.1, 102.2, 103.3], [], [304.4, 305.5]]
    assert awkward1.kernels(three) == "cuda"


def test_add_2():
    one = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], lib="cuda")
    two = 100
    assert awkward1.kernels(one) == "cuda"
    three = one + two
    assert awkward1.to_list(three) == [[101.1, 102.2, 103.3], [], [104.4, 105.5]]
    assert awkward1.kernels(three) == "cuda"
