# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test():
    i = numpy.arange(12, dtype="i4")
    assert sys.getrefcount(i) == 2

    i2 = awkward1.layout.Index(i)
    assert sys.getrefcount(i) == 3

    x = awkward1.layout.Identity(i2, [(0, "hey"), (1, "there")], 1, 2)
    assert sys.getrefcount(i) == 3

    tmp = numpy.asarray(x)
    assert tmp.tolist() == [[0,  1,  2,  3],
                            [4,  5,  6,  7],
                            [8,  9, 10, 11]]
    assert sys.getrefcount(i) == 3

    del tmp
    assert sys.getrefcount(i) == 3
    del i2
    assert sys.getrefcount(i) == 3
    del x
    assert sys.getrefcount(i) == 2
