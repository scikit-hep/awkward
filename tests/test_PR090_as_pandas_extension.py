# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

pandas = pytest.importorskip("pandas")

def test_basic():
    nparray = numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    akarray = awkward1.Array(nparray)
    dfarray = pandas.DataFrame({"x": akarray})

    assert dfarray.x[2] == 2.2

    nparray[2] = 999
    assert dfarray.x[2] == 999
