# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_fromnumpy():
    a = numpy.arange(2*3*5).reshape((2, 3, 5))
    b = awkward1.fromnumpy(a)
    assert awkward1.tolist(a) == awkward1.tolist(b)

def test_highlevel():
    a = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    assert repr(a) == "<Array [[1.1 2.2 3.3] []] ... [6.6] [7.7 8.8 9.9]] type='5 * var * float64'>"
