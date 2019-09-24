# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import pytest
import numpy

import awkward1

def test_listarray():
    starts = awkward1.layout.Index64(numpy.array([0, 3, 3]))
    stops = awkward1.layout.Index64(numpy.array([3, 3, 5]))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    array = awkward1.layout.ListArray64(starts, stops, content)
    assert awkward1.tolist(array) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.tolist(array[1:]) == [[], [4.4, 5.5]]
