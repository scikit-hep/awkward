# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

def test_slice():
    assert repr(awkward1.layout.Slice(3)) == "[3]"
    assert repr(awkward1.layout.Slice(slice(None))) == "[::]"
    assert repr(awkward1.layout.Slice(slice(10))) == "[:10:]"
    assert repr(awkward1.layout.Slice(slice(1, 2))) == "[1:2:]"
    assert repr(awkward1.layout.Slice(slice(1, None))) == "[1::]"
    assert repr(awkward1.layout.Slice(slice(None, None, 2))) == "[::2]"
    assert repr(awkward1.layout.Slice(slice(1, 2, 3))) == "[1:2:3]"
    if not py27:
        assert repr(awkward1.layout.Slice(Ellipsis)) == "[...]"
    assert repr(awkward1.layout.Slice(numpy.newaxis)) == "[newaxis]"
    assert repr(awkward1.layout.Slice(None)) == "[newaxis]"
    assert repr(awkward1.layout.Slice([1, 2, 3])) == "[array([1, 2, 3])]"
    assert repr(awkward1.layout.Slice(numpy.array([[1, 2], [3, 4], [5, 6]]))) == "[array([[1, 2], [3, 4], [5, 6]])]"
    assert repr(awkward1.layout.Slice(numpy.array([1, 2, 3, 4, 5, 6])[::-2])) == "[array([6, 4, 2])]"
    a = numpy.arange(3*5).reshape(3, 5)[1::, ::-2]
    assert repr(awkward1.layout.Slice(a)) == "[array(" + str(a.tolist()) + ")]"
    a = numpy.arange(3*5).reshape(3, 5)[::-1, ::2]
    assert repr(awkward1.layout.Slice(a)) == "[array(" + str(a.tolist()) + ")]"
    assert repr(awkward1.layout.Slice([True, True, False, False, True])) == "[array([0, 1, 4])]"
    assert repr(awkward1.layout.Slice([[True, True], [False, False], [True, False]])) == "[array([0, 0, 2]), array([0, 1, 0])]"
    assert repr(awkward1.layout.Slice(())) == "[]"
    assert repr(awkward1.layout.Slice((3,))) == "[3]"
    assert repr(awkward1.layout.Slice((3, slice(1, 2, 3)))) == "[3, 1:2:3]"
    assert repr(awkward1.layout.Slice((slice(None), [1, 2, 3]))) == "[::, array([1, 2, 3])]"
    assert repr(awkward1.layout.Slice(([1, 2, 3], slice(None)))) == "[array([1, 2, 3]), ::]"
    assert repr(awkward1.layout.Slice((slice(None), [True, True, False, False, True]))) == "[::, array([0, 1, 4])]"
    assert repr(awkward1.layout.Slice((slice(None), [[True, True], [False, False], [True, False]]))) == "[::, array([0, 0, 2]), array([0, 1, 0])]"
    assert repr(awkward1.layout.Slice(([[True, True], [False, False], [True, False]], slice(None)))) == "[array([0, 0, 2]), array([0, 1, 0]), ::]"

    with pytest.raises(ValueError):
        awkward1.layout.Slice(numpy.array([1.1, 2.2, 3.3]))
    with pytest.raises(ValueError):
        awkward1.layout.Slice(numpy.array(["one", "two", "three"]))
    with pytest.raises(ValueError):
        awkward1.layout.Slice(numpy.array([1, 2, 3, None, 4, 5]))

    assert repr(awkward1.layout.Slice((123, [[1, 1], [2, 2], [3, 3]]))) == "[array([[123, 123], [123, 123], [123, 123]]), array([[1, 1], [2, 2], [3, 3]])]"
    assert repr(awkward1.layout.Slice(([[1, 1], [2, 2], [3, 3]], 123))) == "[array([[1, 1], [2, 2], [3, 3]]), array([[123, 123], [123, 123], [123, 123]])]"
    assert repr(awkward1.layout.Slice(([[100, 200, 300, 400]], [[1], [2], [3]]))) == "[array([[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]]), array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])]"
    assert repr(awkward1.layout.Slice(([[1], [2], [3]], [[100, 200, 300, 400]]))) == "[array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]), array([[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]])]"

    with pytest.raises(ValueError):
        awkward1.layout.Slice((3, slice(None), [[1], [2], [3]]))
    with pytest.raises(ValueError):
        awkward1.layout.Slice(([[1, 2, 3, 4]], slice(None), [[1], [2], [3]]))
    with pytest.raises(ValueError):
        awkward1.layout.Slice((slice(None), 3, slice(None), [[1], [2], [3]], slice(None)))
    with pytest.raises(ValueError):
        awkward1.layout.Slice((slice(None), [[1, 2, 3, 4]], slice(None), [[1], [2], [3]], slice(None)))
    assert repr(awkward1.layout.Slice((slice(None), 3, [[1], [2], [3]], slice(None)))) == "[::, array([[3], [3], [3]]), array([[1], [2], [3]]), ::]"
    assert repr(awkward1.layout.Slice((slice(None), [[1, 2, 3, 4]], [[1], [2], [3]], slice(None)))) == "[::, array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]), ::]"

def test_numpyarray_getitem():
    a = numpy.arange(10)
    b = awkward1.layout.NumpyArray(a)
    assert b.getitem(3) == a[3]
    assert b.getitem(-3) == a[-3]
    assert awkward1.tolist(b.getitem(())) == awkward1.tolist(a[()])
    assert awkward1.tolist(b.getitem(slice(None))) == awkward1.tolist(a[slice(None)])
    assert awkward1.tolist(b.getitem(slice(3, 7))) == awkward1.tolist(a[slice(3, 7)])
    assert awkward1.tolist(b.getitem(slice(3, -3))) == awkward1.tolist(a[slice(3, -3)])
    assert awkward1.tolist(b.getitem(slice(1, 7, 2))) == awkward1.tolist(a[slice(1, 7, 2)])
    assert awkward1.tolist(b.getitem(slice(-8, 7, 2))) == awkward1.tolist(a[slice(-8, 7, 2)])
    assert awkward1.tolist(b.getitem(slice(None, 7, 2))) == awkward1.tolist(a[slice(None, 7, 2)])
    assert awkward1.tolist(b.getitem(slice(None, -3, 2))) == awkward1.tolist(a[slice(None, -3, 2)])
    assert awkward1.tolist(b.getitem(slice(8, None, -1))) == awkward1.tolist(a[slice(8, None, -1)])
    assert awkward1.tolist(b.getitem(slice(8, None, -2))) == awkward1.tolist(a[slice(8, None, -2)])
    assert awkward1.tolist(b.getitem(slice(-2, None, -2))) == awkward1.tolist(a[slice(-2, None, -2)])

    # a = numpy.arange(7*5).reshape(7, 5)
    # b = awkward1.layout.NumpyArray(a)
    #
    # print(a[1:4])
    # print(awkward1.tolist(b.getitem(slice(1, 4))))
    # raise Exception
