# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_2d():
    array = awkward1.from_iter([
        [3.3, 2.2, 5.5, 1.1, 4.4],
        [4.4, 2.2, 1.1, 3.3, 5.5],
        [2.2, 1.1, 4.4, 3.3, 5.5]], highlevel=False)
    assert awkward1.to_list(array.argmin(axis=0)) == [2, 2, 1, 0, 0]
    assert awkward1.to_list(array.argmin(axis=1)) == [3, 2, 1]

def test_3d():
    array = awkward1.from_iter([
        [[ 3.3,  2.2,  5.5,  1.1,  4.4],
         [ 4.4,  2.2,  1.1,  3.3,  5.5],
         [ 2.2,  1.1,  4.4,  3.3,  5.5]],
        [[-3.3,  2.2, -5.5,  1.1,  4.4],
         [ 4.4, -2.2,  1.1, -3.3,  5.5],
         [ 2.2,  1.1,  4.4,  3.3, -5.5]]], highlevel=False)
    assert awkward1.to_list(array.argmin(axis=0)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1]]
    assert awkward1.to_list(array.argmin(axis=1)) == [
        [2, 2, 1, 0, 0],
        [0, 1, 0, 1, 2]]
    assert awkward1.to_list(array.argmin(axis=2)) == [
        [3, 2, 1],
        [2, 3, 4]]
    assert awkward1.to_list(array.argmin(axis=-1)) == [
        [3, 2, 1],
        [2, 3, 4]]
    assert awkward1.to_list(array.argmin(axis=-2)) == [
        [2, 2, 1, 0, 0],
        [0, 1, 0, 1, 2]]
    assert awkward1.to_list(array.argmin(axis=-3)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1]]

def test_jagged():
    array = awkward1.from_iter([[2.2, 1.1, 3.3], [], [4.4, 5.5], [5.5], [-4.4, -5.5, -6.6]], highlevel=False)
    assert awkward1.to_list(array.argmin(axis=1)) == [1, None, 0, 0, 2]

    index2 = awkward1.layout.Index64(numpy.array([4, 3, 2, 1, 0], dtype=numpy.int64))
    array2 = awkward1.layout.IndexedArray64(index2, array)
    assert awkward1.to_list(array2.argmin(axis=1)) == [2, 0, 0, None, 1]

    index3 = awkward1.layout.Index64(numpy.array([4, 3, -1, 4, 0], dtype=numpy.int64))
    array2 = awkward1.layout.IndexedArray64(index3, array)
    assert awkward1.to_list(array2.argmin(axis=1)) == [2, 0, None, 2, 1]
    assert awkward1.to_list(array2.argmin(axis=-1)) == [2, 0, None, 2, 1]

def test_missing():
    array = awkward1.from_iter([[[2.2, 1.1, 3.3]], [[]], [None, None, None], [[-4.4, -5.5, -6.6]]], highlevel=False)
    assert awkward1.to_list(array.argmin(axis=2)) == [[1], [None], [None, None, None], [2]]

def test_highlevel():
    array = awkward1.Array([
        [3.3, 1.1,  5.5, 1.1, 4.4],
        [4.4, 2.2,  1.1, 6.6     ],
        [2.2, 3.3, -1.1          ]])
    assert awkward1.argmin(array) == 11
    assert awkward1.argmax(array) == 8
    assert awkward1.to_list(awkward1.argmin(array, axis=0)) == [2, 0, 2, 0, 0]
    assert awkward1.to_list(awkward1.argmax(array, axis=0)) == [1, 2, 0, 1, 0]
    assert awkward1.to_list(awkward1.argmin(array, axis=1)) == [1, 2, 2]
    assert awkward1.to_list(awkward1.argmax(array, axis=1)) == [2, 3, 1]
