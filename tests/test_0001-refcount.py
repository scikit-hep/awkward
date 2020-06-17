# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import numpy

import awkward1

def test_refcount():
    o = numpy.arange(10, dtype="i4")
    c = numpy.arange(12).reshape(3, 4)

    for order in itertools.permutations(["del i, n", "del l", "del l2"]):
        i = awkward1.layout.Index32(o)
        n = awkward1.layout.NumpyArray(c)
        l = awkward1.layout.ListOffsetArray32(i, n)
        l2 = awkward1.layout.ListOffsetArray32(i, l)

        for statement in order:
            assert sys.getrefcount(o), sys.getrefcount(c) == (3, 3)
            exec(statement)
            assert sys.getrefcount(o), sys.getrefcount(c) == (2, 2)
