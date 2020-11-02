# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test():
    one_content = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9, 10.0]]).layout
    one_starts = awkward1.layout.Index64(numpy.array([0, 2, 3, 3], dtype=numpy.int64))
    one_stops  = awkward1.layout.Index64(numpy.array([2, 3, 3, 5], dtype=numpy.int64))
    one = awkward1.layout.ListArray64(one_starts, one_stops, one_content)
    assert awkward1.to_list(one) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9, 10.0]]]

    two_content = awkward1.Array([[123], [1.1, 2.2, 3.3], [], [234], [4.4, 5.5], [345], [6.6], [7.7, 8.8, 9.9, 10.0], [456]]).layout
    two_starts = awkward1.layout.Index64(numpy.array([1, 4, 5, 6], dtype=numpy.int64))
    two_stops  = awkward1.layout.Index64(numpy.array([3, 5, 5, 8], dtype=numpy.int64))
    two = awkward1.layout.ListArray64(two_starts, two_stops, two_content)
    assert awkward1.to_list(two) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9, 10.0]]]

    assert awkward1.to_list(one[[[[0, 1, 2], []], [[0, 1]], [], [[0], [0, 1, 2, 3]]]]) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9, 10.0]]]
    assert awkward1.to_list(two[[[[0, 1, 2], []], [[0, 1]], [], [[0], [0, 1, 2, 3]]]]) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9, 10.0]]]
