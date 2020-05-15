# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy

import awkward1

def test_identity():
    a = numpy.arange(10)
    b = awkward1.layout.NumpyArray(a)
    b.setidentities()
    assert numpy.array(b.identities).tolist() == numpy.arange(10).reshape(-1, 1).tolist()

    assert numpy.array(b[3]) == a[3]
    assert numpy.array(b[3:7].identities).tolist() == numpy.arange(10).reshape(-1, 1)[3:7].tolist()
    assert numpy.array(b[[7, 3, 3, -4]].identities).tolist() == numpy.arange(10).reshape(-1, 1)[[7, 3, 3, -4]].tolist()
    assert numpy.array(b[[True, True, True, False, False, False, True, False, True, False]].identities).tolist() == numpy.arange(10).reshape(-1, 1)[[True, True, True, False, False, False, True, False, True, False]].tolist()

    assert numpy.array(b[1:][3]) == a[1:][3]
    assert numpy.array(b[1:][3:7].identities).tolist() == numpy.arange(10).reshape(-1, 1)[1:][3:7].tolist()
    assert numpy.array(b[1:][[7, 3, 3, -4]].identities).tolist() == numpy.arange(10).reshape(-1, 1)[1:][[7, 3, 3, -4]].tolist()
    assert numpy.array(b[1:][[True, True, False, False, False, True, False, True, False]].identities).tolist() == numpy.arange(10).reshape(-1, 1)[1:][[True, True, False, False, False, True, False, True, False]].tolist()
