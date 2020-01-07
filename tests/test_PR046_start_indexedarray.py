# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_basic():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = awkward1.layout.Index64(numpy.array([2, 2, 0, 3, 4], dtype=numpy.int64))
    array = awkward1.layout.IndexedArray64(index, content)

    assert awkward1.tolist(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
