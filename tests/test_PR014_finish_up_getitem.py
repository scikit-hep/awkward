# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

content = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(-1, 7))
offsets1 = awkward1.layout.Index64(numpy.arange(0, 2*3*5 + 5, 5))
offsets2 = awkward1.layout.Index64(numpy.arange(0, 2*3 + 3, 3))
listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content)
model1 = numpy.arange(2*3*5*7).reshape(2*3, 5, 7)
listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, listoffsetarray1)
model2 = numpy.arange(2*3*5*7).reshape(2, 3, 5, 7)

def test_basic():
    assert awkward1.tolist(model1) == awkward1.tolist(listoffsetarray1)
    assert awkward1.tolist(model2) == awkward1.tolist(listoffsetarray2)
