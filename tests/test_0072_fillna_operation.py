# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_fillna_empty_array():
    empty = awkward1.layout.EmptyArray()

    assert awkward1.tolist(empty) == []

def test_fillna_numpy_array():
    content1 = awkward1.layout.NumpyArray(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
    content2 = content1.rpad(3,0)
    assert awkward1.tolist(content2.fillna(0)) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], 0]
