# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import pytest
import numpy

import awkward1

def test_identity():
    a = numpy.arange(10)
    b = awkward1.layout.NumpyArray(a)
    b.setid()
    assert numpy.array(b.id).tolist() == numpy.arange(10).reshape(-1, 1).tolist()
    assert numpy.array(b[3]) == 3
    assert numpy.array(b[3:7].id).tolist() == numpy.arange(3, 7).reshape(-1, 1).tolist()
    assert numpy.array(b[[7, 3, 3, -4]].id).tolist() == numpy.arange(10).reshape(-1, 1)[[7, 3, 3, -4]].tolist()
    assert numpy.array(b[[True, True, True, False, False, False, True, False, True, False]].id).tolist() == numpy.arange(10).reshape(-1, 1)[[True, True, True, False, False, False, True, False, True, False]].tolist()
