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

def test_numpyarray_getitem_bystrides():
    a = numpy.arange(10)
    b = awkward1.layout.NumpyArray(a)
    assert b[3] == a[3]
    assert b[-3] == a[-3]
    assert awkward1.tolist(b[()]) == awkward1.tolist(a[()])
    assert awkward1.tolist(b[slice(None)]) == awkward1.tolist(a[slice(None)])
    assert awkward1.tolist(b[slice(3, 7)]) == awkward1.tolist(a[slice(3, 7)])
    assert awkward1.tolist(b[slice(3, 100)]) == awkward1.tolist(a[slice(3, 100)])
    assert awkward1.tolist(b[slice(-100, 7)]) == awkward1.tolist(a[slice(-100, 7)])
    assert awkward1.tolist(b[slice(3, -3)]) == awkward1.tolist(a[slice(3, -3)])
    assert awkward1.tolist(b[slice(1, 7, 2)]) == awkward1.tolist(a[slice(1, 7, 2)])
    assert awkward1.tolist(b[slice(-8, 7, 2)]) == awkward1.tolist(a[slice(-8, 7, 2)])
    assert awkward1.tolist(b[slice(None, 7, 2)]) == awkward1.tolist(a[slice(None, 7, 2)])
    assert awkward1.tolist(b[slice(None, -3, 2)]) == awkward1.tolist(a[slice(None, -3, 2)])
    assert awkward1.tolist(b[slice(8, None, -1)]) == awkward1.tolist(a[slice(8, None, -1)])
    assert awkward1.tolist(b[slice(8, None, -2)]) == awkward1.tolist(a[slice(8, None, -2)])
    assert awkward1.tolist(b[slice(-2, None, -2)]) == awkward1.tolist(a[slice(-2, None, -2)])

    a = numpy.arange(7*5).reshape(7, 5)
    b = awkward1.layout.NumpyArray(a)

    assert awkward1.tolist(b[()]) == awkward1.tolist(a[()])
    assert awkward1.tolist(b[3]) == awkward1.tolist(a[3])
    assert awkward1.tolist(b[(3, 2)]) == awkward1.tolist(a[3, 2])
    assert awkward1.tolist(b[slice(1, 4)]) == awkward1.tolist(a[slice(1, 4)])
    assert awkward1.tolist(b[(3, slice(1, 4))]) == awkward1.tolist(a[3, slice(1, 4)])
    assert awkward1.tolist(b[(slice(1, 4), 3)]) == awkward1.tolist(a[slice(1, 4), 3])
    assert awkward1.tolist(b[(slice(1, 4), slice(2, None))]) == awkward1.tolist(a[slice(1, 4), slice(2, None)])
    assert awkward1.tolist(b[(slice(None, None, -1), slice(2, None))]) == awkward1.tolist(a[slice(None, None, -1), slice(2, None)])

    assert awkward1.tolist(b[:, numpy.newaxis, :]) == awkward1.tolist(a[:, numpy.newaxis, :])
    assert awkward1.tolist(b[numpy.newaxis, :, numpy.newaxis, :, numpy.newaxis]) == awkward1.tolist(a[numpy.newaxis, :, numpy.newaxis, :, numpy.newaxis])

    if not py27:
        assert awkward1.tolist(b[Ellipsis, 3]) == awkward1.tolist(a[Ellipsis, 3])
        assert awkward1.tolist(b[Ellipsis, 3, 2]) == awkward1.tolist(a[Ellipsis, 3, 2])
        assert awkward1.tolist(b[3, Ellipsis]) == awkward1.tolist(a[3, Ellipsis])
        assert awkward1.tolist(b[3, 2, Ellipsis]) == awkward1.tolist(a[3, 2, Ellipsis])

def test_numpyarray_contiguous():
    a = numpy.arange(10)[8::-2]
    b = awkward1.layout.NumpyArray(a)

    assert awkward1.tolist(b) == awkward1.tolist(a)
    assert awkward1.tolist(b.contiguous()) == awkward1.tolist(a)
    b.become_contiguous()
    assert awkward1.tolist(b) == awkward1.tolist(a)

    a = numpy.arange(7*5).reshape(7, 5)[::-1, ::2]
    b = awkward1.layout.NumpyArray(a)

    assert awkward1.tolist(b) == awkward1.tolist(a)
    assert awkward1.tolist(b.contiguous())
    b.become_contiguous()
    assert awkward1.tolist(b) == awkward1.tolist(a)

def test_numpyarray_getitem_next():
    a = numpy.arange(10)
    b = awkward1.layout.NumpyArray(a)
    c = numpy.array([7, 3, 3, 5])
    assert awkward1.tolist(b[c]) == awkward1.tolist(a[c])

    a = numpy.arange(10*3).reshape(10, 3)
    b = awkward1.layout.NumpyArray(a)
    c = numpy.array([7, 3, 3, 5])
    assert awkward1.tolist(b[c]) == awkward1.tolist(a[c])

    a = numpy.arange(7*5).reshape(7, 5)
    b = awkward1.layout.NumpyArray(a)
    c1 = numpy.array([4, 1, 1, 3])
    c2 = numpy.array([2, 2, 0, 1])
    assert awkward1.tolist(b[c1, c2]) == awkward1.tolist(a[c1, c2])
    c1 = numpy.array([[4, 1], [1, 3], [0, 4]])
    c2 = numpy.array([[2, 2], [0, 1], [1, 3]])
    assert awkward1.tolist(b[c1, c2]) == awkward1.tolist(a[c1, c2])
