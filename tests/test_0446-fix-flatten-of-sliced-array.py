# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    array = awkward1.Array([[1, 2, 3], [], [4, 5], [6, 7, 8, 9]])
    assert awkward1.flatten(array[:-1], axis=1).tolist() == [1, 2, 3, 4, 5]
    assert awkward1.flatten(array[:-2], axis=1).tolist() == [1, 2, 3]
    assert awkward1.flatten(array[:-1], axis=None).tolist() == [1, 2, 3, 4, 5]
    assert awkward1.flatten(array[:-2], axis=None).tolist() == [1, 2, 3]
