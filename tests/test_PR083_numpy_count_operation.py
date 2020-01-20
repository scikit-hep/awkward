# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def count(data, axis=0):
    if axis < 0:
        raise NotImplementedError("axis < 0 is much harder for untyped data...")
    if isinstance(data, list):
        if axis == 0:
            if all(isinstance(x, list) for x in data):
                return [len(x) for x in data]
            else:
                raise ValueError("cannot count the lengths of non-lists")
        else:
            return [count(x, axis - 1) for x in data]
    else:
        raise ValueError("cannot count {0} objects".format(type(data)))

def test_count_numpy_array():
    array = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5))
    assert awkward1.tolist(array) == [[[ 0,  1,  2,  3,  4], [ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]],
                                      [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert awkward1.tolist(array.count()) == [3, 3]
    assert count(awkward1.tolist(array)) == [3, 3]

    assert awkward1.tolist(array.flatten()) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
    assert awkward1.tolist(array.flatten().count()) == [5, 5, 5, 5, 5, 5]
    assert count(awkward1.tolist(array), 1) == [[5, 5, 5], [5, 5, 5]]
    assert awkward1.tolist(array.count(1)) == [[5, 5, 5], [5, 5, 5]]

    assert awkward1.tolist(array.flatten(1)) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    assert awkward1.tolist(array.flatten(1).count()) == [15, 15]
    ### ValueError assert count(awkward1.tolist(array), 2) == [[5, 5, 5], [5, 5, 5]]
    with pytest.raises(ValueError) :
        assert awkward1.tolist(array.count(2)) == []
