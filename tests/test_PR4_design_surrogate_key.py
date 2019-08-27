# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import gc

import pytest
import numpy

import awkward1

def test():
    i = numpy.arange(12, dtype="i4").reshape(3, 4)
    assert sys.getrefcount(i) == 2

    i2 = awkward1.layout.Identity(awkward1.layout.Identity.newref(), [(0, "hey"), (1, "there")], 1, 2, i)
    assert sys.getrefcount(i) == 3

    tmp = numpy.asarray(i2)
    assert tmp.tolist() == [[0,  1,  2,  3],
                            [4,  5,  6,  7],
                            [8,  9, 10, 11]]

    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 3)

    del tmp
    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2)

    del i2
    assert sys.getrefcount(i) == 2

# def test_refcount():
#     i = numpy.arange(10, dtype="i4")
#     i2 = awkward1.layout.Index(i)
#     i3 = awkward1.layout.Identity(i2, [], 0, 2)
#     x = numpy.arange(12).reshape(3, 4)
#     x2 = awkward1.layout.NumpyArray(x)
#     x2.id = i3
#     del i
#     del i2
#     del i3
#     del x
#     i4 = x2.id
#     del x2
#     gc.collect()
#     assert numpy.asarray(i4).tolist() == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
#     del i4
#     gc.collect()
