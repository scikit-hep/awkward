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

def test_listoffsetarray1():
    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 4, -5), depth):
            assert awkward1.tolist(model1[cuts]) == awkward1.tolist(listoffsetarray1[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(None), slice(1, None), slice(None, -1), slice(None, None, 2)), depth):
            assert awkward1.tolist(model1[cuts]) == awkward1.tolist(listoffsetarray1[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(1, None), slice(None, -1), 2, -2), depth):
            assert awkward1.tolist(model1[cuts]) == awkward1.tolist(listoffsetarray1[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], 2, -2), depth):
            assert awkward1.tolist(model1[cuts]) == awkward1.tolist(listoffsetarray1[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], slice(1, None), slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.tolist(model1[cuts]) == awkward1.tolist(listoffsetarray1[cuts])
