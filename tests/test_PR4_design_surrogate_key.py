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

    tmp2 = i2.array
    assert tmp2.tolist() == [[0,  1,  2,  3],
                             [4,  5,  6,  7],
                             [8,  9, 10, 11]]

    assert (sys.getrefcount(i), sys.getrefcount(i2)) == (3, 2 + 1*py27)

    del tmp2
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

def test_listoffsetarray_setid():
    content = awkward1.layout.NumpyArray(numpy.arange(10))
    offsets = awkward1.layout.Index(numpy.array([0, 3, 3, 5, 10], dtype="i4"))
    jagged = awkward1.layout.ListOffsetArray(offsets, content)
    jagged.setid()
    assert numpy.asarray(jagged.id).tolist() == [[0], [1], [2], [3]]
    assert numpy.asarray(jagged.content.id).tolist() == [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]]
    assert numpy.asarray(jagged.content[3:7].id).tolist() == [[2, 0], [2, 1], [3, 0], [3, 1]]
    assert numpy.asarray(jagged[0].id).tolist() == [[0, 0], [0, 1], [0, 2]]
    assert numpy.asarray(jagged[1].id).tolist() == []
    assert numpy.asarray(jagged[2].id).tolist() == [[2, 0], [2, 1]]
    assert numpy.asarray(jagged[3].id).tolist() == [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4]]
    assert numpy.asarray(jagged[1:3].id).tolist() == [[1], [2]]

def test_setid_none():
    offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    array = awkward1.layout.ListOffsetArray(offsets, content)
    assert array.id is None
    assert array.content.id is None
    array.id = None
    assert array.id is None
    assert array.content.id is None
    repr(array)
    array.setid()
    repr(array)
    assert array.id is not None
    assert array.content.id is not None
    array.id = None
    assert array.id is None
    assert array.content.id is None

def test_setid_constructor():
    offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], "i4"))
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]), id=awkward1.layout.Identity(awkward1.layout.Identity.newref(), [], 0, 2, numpy.array([[0, 0], [0, 1], [2, 0]], dtype="i4")))
    array = awkward1.layout.ListOffsetArray(offsets, content, id=awkward1.layout.Identity(awkward1.layout.Identity.newref(), [], 0, 1, numpy.array([[0], [1], [2]], dtype="i4")))
    assert numpy.asarray(array.id).tolist() == [[0], [1], [2]]
    assert numpy.asarray(array.content.id).tolist() == [[0, 0], [0, 1], [2, 0]]
