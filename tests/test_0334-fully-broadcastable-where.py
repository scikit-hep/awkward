# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    one = awkward1.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]])
    two = awkward1.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
    condition = awkward1.Array([[False, True, False], [], [True, False], [True], [False, False, True, True]])
    assert awkward1.where(condition, one, two).tolist() == [[0, 1, 2.2], [], [3, 4.4], [5], [6.6, 7.7, 8, 9]]


def test_issue_334():
    a = awkward1.Array([1, 2, 3, 4])
    b = awkward1.Array([-1])
    c = awkward1.Array([True, False, True, True])

    assert awkward1.where(c, a, b).tolist() == [1, -1, 3, 4]
    assert awkward1.where(*awkward1.broadcast_arrays(c, a, b)).tolist() == [1, -1, 3, 4]
    assert awkward1.where(c, a, -1).tolist() == [1, -1, 3, 4]
    assert awkward1.where(*awkward1.broadcast_arrays(c, a, -1)).tolist() == [1, -1, 3, 4]
