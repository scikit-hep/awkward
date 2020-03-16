# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_2d():
    array = awkward1.Array([
        [3.3, 2.2, 5.5, 1.1, 4.4],
        [4.4, 2.2, 1.1, 3.3, 5.5],
        [2.2, 1.1, 4.4, 3.3, 5.5]]).layout
    assert awkward1.tolist(array.argmin(axis=0)) == [2, 2, 1, 0, 0]
    assert awkward1.tolist(array.argmin(axis=1)) == [3, 2, 1]

def test_3d():
    array = awkward1.Array([
        [[ 3.3,  2.2,  5.5,  1.1,  4.4],
         [ 4.4,  2.2,  1.1,  3.3,  5.5],
         [ 2.2,  1.1,  4.4,  3.3,  5.5]],
        [[-3.3,  2.2, -5.5,  1.1,  4.4],
         [ 4.4, -2.2,  1.1, -3.3,  5.5],
         [ 2.2,  1.1,  4.4,  3.3, -5.5]]]).layout
    assert awkward1.tolist(array.argmin(axis=0)) == [
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1]]
    assert awkward1.tolist(array.argmin(axis=1)) == [
        [2, 2, 1, 0, 0],
        [0, 1, 0, 1, 2]]
    assert awkward1.tolist(array.argmin(axis=2)) == [
        [3, 2, 1],
        [2, 3, 4]]
