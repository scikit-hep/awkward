# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_array_3d():
    array = awkward1.Array(numpy.arange(3*5*2).reshape(3, 5, 2))
    assert awkward1.to_list(array) ==  [[[ 0,  1], [ 2,  3], [ 4,  5], [ 6,  7], [ 8,  9]],
                                        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
                                        [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]]]
    assert str(awkward1.type(array)) == '3 * 5 * 2 * int64'
    assert awkward1.num(array, axis=0) == 3
    assert awkward1.to_list(awkward1.num(array, axis=1)) == [5, 5, 5]
    assert str(awkward1.type(awkward1.num(array, axis=1))) == '3 * int64'
    assert awkward1.to_list(awkward1.num(array, axis=2)) == [[2, 2, 2, 2, 2],
                                                             [2, 2, 2, 2, 2],
                                                             [2, 2, 2, 2, 2]]
    assert str(awkward1.type(awkward1.num(array, axis=2))) == '3 * 5 * int64'
    with pytest.raises(ValueError) as err:
        assert awkward1.num(array, axis=3)
    assert str(err.value) == "'axis' out of range for 'num'"

    assert awkward1.to_list(awkward1.num(array, axis=-1)) == [[2, 2, 2, 2, 2],
                                                              [2, 2, 2, 2, 2],
                                                              [2, 2, 2, 2, 2]]
    assert str(awkward1.type(awkward1.num(array, axis=-1))) == '3 * 5 * int64'
    assert awkward1.to_list(awkward1.num(array, axis=-2)) == [5, 5, 5]
    assert str(awkward1.type(awkward1.num(array, axis=-2))) == '3 * int64'
    assert awkward1.num(array, axis=-3) == 3

    with pytest.raises(ValueError) as err:
        assert awkward1.num(array, axis=-4)
    assert str(err.value) == "'axis' out of range for 'num'"
