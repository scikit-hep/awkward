# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import pytest
import numpy

import awkward1

def test_listarray():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    starts1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6]))
    stops1  = awkward1.layout.Index64(numpy.array([3, 3, 5, 6, 9]))
    array1  = awkward1.layout.ListArray64(starts1, stops1, content)
    starts2 = awkward1.layout.Index64(numpy.array([0, 2, 3, 3]))
    stops2  = awkward1.layout.Index64(numpy.array([2, 3, 3, 5]))
    array2  = awkward1.layout.ListArray64(starts2, stops2, array1)

    assert awkward1.tolist(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(array1[2]) == [4.4, 5.5]
    assert awkward1.tolist(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert awkward1.tolist(array2) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert awkward1.tolist(array2[1]) == [[4.4, 5.5]]
    assert awkward1.tolist(array2[1:-1]) == [[[4.4, 5.5]], []]

    assert awkward1.tolist(array1[numpy.array([2, 0, 0, 1, -1])]) == [[4.4, 5.5], [1.1, 2.2, 3.3], [1.1, 2.2, 3.3], [], [7.7, 8.8, 9.9]]
    # assert awkward1.tolist(array1[numpy.array([2, 0, 0, -1]), numpy.array([1, 1, 0, 0])]) == [5.5, 2.2, 1.1, 7.7]
