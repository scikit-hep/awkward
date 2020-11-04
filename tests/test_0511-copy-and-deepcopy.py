# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import copy

import pytest

import numpy
import awkward1


def test():
    np_array = numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    one = awkward1.Array(np_array)

    np_array[1] = 999
    assert awkward1.to_list(one) == [0.0, 999, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    two = copy.copy(one)
    np_array[3] = 123
    assert awkward1.to_list(two) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    three = copy.deepcopy(two)
    four = numpy.copy(two)
    np_array[5] = 321
    assert awkward1.to_list(three) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert awkward1.to_list(four) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    assert awkward1.to_list(copy.deepcopy(awkward1.Array([[1, 2, 3], [], [4, 5]]))) == [[1, 2, 3], [], [4, 5]]

    assert awkward1.to_list(copy.deepcopy(awkward1.Record({"one": 1, "two": 2.2}))) == awkward1.to_list(copy.deepcopy(awkward1.Record({"one": 1, "two": 2.2})))
