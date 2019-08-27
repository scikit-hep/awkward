# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import gc

import pytest
import numpy

import awkward1

py27 = 2 if sys.version_info[0] < 3 else 1

def test_refcount1():
    i = numpy.arange(12, dtype="i4").reshape(3, 4)
    assert sys.getrefcount(i) == 2

    i2 = awkward1.layout.Identity(awkward1.layout.Identity.newref(), [(0, "hey"), (1, "there")], 1, 2, i)
    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2)

    tmp = numpy.asarray(i2)
    assert tmp.tolist() == [[0,  1,  2,  3],
                            [4,  5,  6,  7],
                            [8,  9, 10, 11]]

    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2 + 1*py27)

    del tmp
    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2)

    del i2
    assert sys.getrefcount(i) == 2

def test_refcount2():
    i = numpy.arange(6, dtype="i4").reshape(3, 2)
    i2 = awkward1.layout.Identity(awkward1.layout.Identity.newref(), [], 0, 2, i)
    x = numpy.arange(12).reshape(3, 4)
    x2 = awkward1.layout.NumpyArray(x)
    x2.id = i2
    del i
    del i2
    del x
    i3 = x2.id
    del x2
    gc.collect()
    assert numpy.asarray(i3).tolist() == [[0, 1], [2, 3], [4, 5]]
    del i3
    gc.collect()

def test_refcount3():
    i = numpy.arange(6, dtype="i4").reshape(3, 2)
    i2 = awkward1.layout.Identity(awkward1.layout.Identity.newref(), [], 0, 2, i)
    x = numpy.arange(12).reshape(3, 4)
    x2 = awkward1.layout.NumpyArray(x)
    x2.id = i2
    del i2
    assert sys.getrefcount(i) == 3
    x2.id = None
    assert sys.getrefcount(i) == 2

def test_numpyarray_setid():
    x = numpy.arange(160).reshape(40, 4)
    x2 = awkward1.layout.NumpyArray(x)
    x2.setid()
    assert numpy.asarray(x2.id).tolist() == numpy.arange(40).reshape(40, 1).tolist()
