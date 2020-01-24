# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_mixing_lists_and_none():
    def add(a, b):
        outer = []
        for ai, bi in zip(a, b):
            if ai is None or bi is None:
                outer.append(None)
            else:
                inner = []
                for aj, bj in zip(ai, bi):
                    if aj is None or bj is None:
                        inner.append(None)
                    else:
                        inner.append(aj + bj)
                outer.append(inner)
        return outer

    a00 = awkward1.Array([[1.1,  2.2, 3.3],   [], [4.4, 5.5], [ 6.6, 7.7, 8.8, 9.9]])
    a01 = awkward1.Array([[1.1, None, 3.3],   [], [4.4, 5.5], [ 6.6, 7.7, 8.8, 9.9]])
    a02 = awkward1.Array([[1.1, None, 3.3],   [], [4.4, 5.5], [None, 7.7, 8.8, 9.9]])
    a10 = awkward1.Array([[1.1,  2.2, 3.3],   [],       None, [ 6.6, 7.7, 8.8, 9.9]])
    a11 = awkward1.Array([[1.1, None, 3.3],   [],       None, [ 6.6, 7.7, 8.8, 9.9]])
    a12 = awkward1.Array([[1.1, None, 3.3],   [],       None, [None, 7.7, 8.8, 9.9]])
    a20 = awkward1.Array([[1.1,  2.2, 3.3], None,       None, [ 6.6, 7.7, 8.8, 9.9]])
    a21 = awkward1.Array([[1.1, None, 3.3], None,       None, [ 6.6, 7.7, 8.8, 9.9]])
    a22 = awkward1.Array([[1.1, None, 3.3], None,       None, [None, 7.7, 8.8, 9.9]])

    b00 = awkward1.Array([[100,  200, 300],   [], [400, 500], [ 600, 700, 800, 900]])
    b01 = awkward1.Array([[100, None, 300],   [], [400, 500], [ 600, 700, 800, 900]])
    b02 = awkward1.Array([[100, None, 300],   [], [400, 500], [None, 700, 800, 900]])
    b10 = awkward1.Array([[100,  200, 300],   [],       None, [ 600, 700, 800, 900]])
    b11 = awkward1.Array([[100, None, 300],   [],       None, [ 600, 700, 800, 900]])
    b12 = awkward1.Array([[100, None, 300],   [],       None, [None, 700, 800, 900]])
    b20 = awkward1.Array([[100,  200, 300], None,       None, [ 600, 700, 800, 900]])
    b21 = awkward1.Array([[100, None, 300], None,       None, [ 600, 700, 800, 900]])
    b22 = awkward1.Array([[100, None, 300], None,       None, [None, 700, 800, 900]])

    for a in (a00, a01, a02, a10, a11, a12, a20, a21, a22):
        for b in (b00, b01, b02, b10, b11, b12, b20, b21, b22):
            assert awkward1.tolist(a + b) == add(a, b)

def test_explicit_broadcasting():
    nparray = numpy.arange(2*3*5).reshape(2, 3, 5)
    lsarray = awkward1.Array(nparray.tolist())
    rgarray = awkward1.Array(nparray)

    # explicit left-broadcasting
    assert awkward1.tolist(rgarray + numpy.array([[[100]], [[200]]])) == awkward1.tolist(nparray + numpy.array([[[100]], [[200]]]))
    assert awkward1.tolist(lsarray + numpy.array([[[100]], [[200]]])) == awkward1.tolist(nparray + numpy.array([[[100]], [[200]]]))
    assert awkward1.tolist(numpy.array([[[100]], [[200]]]) + rgarray) == awkward1.tolist(numpy.array([[[100]], [[200]]]) + nparray)
    assert awkward1.tolist(numpy.array([[[100]], [[200]]]) + lsarray) == awkward1.tolist(numpy.array([[[100]], [[200]]]) + nparray)

    # explicit right-broadcasting
    assert awkward1.tolist(rgarray + numpy.array([[[100, 200, 300, 400, 500]]])) == awkward1.tolist(nparray + numpy.array([[[100, 200, 300, 400, 500]]]))
    assert awkward1.tolist(lsarray + numpy.array([[[100, 200, 300, 400, 500]]])) == awkward1.tolist(nparray + numpy.array([[[100, 200, 300, 400, 500]]]))
    assert awkward1.tolist(numpy.array([[[100, 200, 300, 400, 500]]]) + rgarray) == awkward1.tolist(numpy.array([[[100, 200, 300, 400, 500]]]) + nparray)
    assert awkward1.tolist(numpy.array([[[100, 200, 300, 400, 500]]]) + lsarray) == awkward1.tolist(numpy.array([[[100, 200, 300, 400, 500]]]) + nparray)

def test_implicit_broadcasting():
    nparray = numpy.arange(2*3*5).reshape(2, 3, 5)
    lsarray = awkward1.Array(nparray.tolist())
    rgarray = awkward1.Array(nparray)

    assert awkward1.tolist(rgarray + numpy.array([100, 200, 300, 400, 500])) == awkward1.tolist(nparray + numpy.array([100, 200, 300, 400, 500]))
    assert awkward1.tolist(numpy.array([100, 200, 300, 400, 500]) + rgarray) == awkward1.tolist(numpy.array([100, 200, 300, 400, 500]) + nparray)

    assert awkward1.tolist(lsarray + numpy.array([100, 200])) == awkward1.tolist(nparray + numpy.array([[[100]], [[200]]]))
    assert awkward1.tolist(numpy.array([100, 200]) + lsarray) == awkward1.tolist(numpy.array([[[100]], [[200]]]) + nparray)

def test_records():
    array = awkward1.Array([[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}], [{"x": 100, "y": 200}]])
    record = array[-1, -1]

    assert awkward1.tolist(array[0] + record) == [{"x": 101, "y": 201.1}, {"x": 102, "y": 202.2}, {"x": 103, "y": 203.3}]

    assert awkward1.tolist(array + record) == [[{"x": 101, "y": 201.1}, {"x": 102, "y": 202.2}, {"x": 103, "y": 203.3}], [], [{"x": 104, "y": 204.4}, {"x": 105, "y": 205.5}], [{"x": 200, "y": 400.0}]]

    assert awkward1.tolist(record + record) == {"x": 200, "y": 400}

def test_tonumpy():
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])), numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))), numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])), numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array(numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))), numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array(["one", "two", "three"])), numpy.array(["one", "two", "three"]))
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array([b"one", b"two", b"three"])), numpy.array([b"one", b"two", b"three"]))
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array([])), numpy.array([]))

    content0 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.float64))
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64))
    tags = awkward1.layout.Index8(numpy.array([0, 1, 1, 0, 0, 0, 1, 0], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 3, 2, 4], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content0, content1]))
    assert numpy.array_equal(awkward1.tonumpy(array), numpy.array([1.1, 1, 2, 2.2, 3.3, 4.4, 3, 5.5]))

    assert awkward1.tonumpy(awkward1.Array([1.1, 2.2, None, None, 3.3])).tolist() == [1.1, 2.2, None, None, 3.3]
    assert awkward1.tonumpy(awkward1.Array([[1.1, 2.2], [None, None], [3.3, 4.4]])).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]
    assert awkward1.tonumpy(awkward1.Array([[1.1, 2.2], None, [3.3, 4.4]])).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]

    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])), numpy.array([(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("x", numpy.int64), ("y", numpy.float64)]))
    assert numpy.array_equal(awkward1.tonumpy(awkward1.Array([(1, 1.1), (2, 2.2), (3, 3.3)])), numpy.array([(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("0", numpy.int64), ("1", numpy.float64)]))

def test_numpy_array():
    assert numpy.array_equal(numpy.asarray(awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])), numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert numpy.array_equal(numpy.asarray(awkward1.Array(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))), numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert numpy.array_equal(numpy.asarray(awkward1.Array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])), numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))
    assert numpy.array_equal(numpy.asarray(awkward1.Array(numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))), numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))
    assert numpy.array_equal(numpy.asarray(awkward1.Array(["one", "two", "three"])), numpy.array(["one", "two", "three"]))
    assert numpy.array_equal(numpy.asarray(awkward1.Array([b"one", b"two", b"three"])), numpy.array([b"one", b"two", b"three"]))
    assert numpy.array_equal(numpy.asarray(awkward1.Array([])), numpy.array([]))

    content0 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.float64))
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3], dtype=numpy.int64))
    tags = awkward1.layout.Index8(numpy.array([0, 1, 1, 0, 0, 0, 1, 0], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 3, 2, 4], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content0, content1]))
    assert numpy.array_equal(numpy.asarray(array), numpy.array([1.1, 1, 2, 2.2, 3.3, 4.4, 3, 5.5]))

    assert numpy.ma.asarray(awkward1.Array([1.1, 2.2, None, None, 3.3])).tolist() == [1.1, 2.2, None, None, 3.3]
    assert numpy.ma.asarray(awkward1.Array([[1.1, 2.2], [None, None], [3.3, 4.4]])).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]
    assert numpy.ma.asarray(awkward1.Array([[1.1, 2.2], None, [3.3, 4.4]])).tolist() == [[1.1, 2.2], [None, None], [3.3, 4.4]]

    assert numpy.array_equal(numpy.asarray(awkward1.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])), numpy.array([(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("x", numpy.int64), ("y", numpy.float64)]))
    assert numpy.array_equal(numpy.asarray(awkward1.Array([(1, 1.1), (2, 2.2), (3, 3.3)])), numpy.array([(1, 1.1), (2, 2.2), (3, 3.3)], dtype=[("0", numpy.int64), ("1", numpy.float64)]))

def test_where():
    one = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    two = awkward1.Array([  0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    condition = awkward1.Array([False, False, False, False, False, True, False, True, False, True])

    assert isinstance(awkward1.where(condition)[0], awkward1.Array)
    assert awkward1.tolist(awkward1.where(condition)[0]) == [5, 7, 9]

    assert awkward1.tolist(awkward1.where(condition, one, two)) == awkward1.tolist(numpy.where(numpy.asarray(condition), numpy.asarray(one), numpy.asarray(two)))

def test_string_equal():
    one = awkward1.Array(["one", "two", "three"])
    two = awkward1.Array(["ONE", "two", "four"])
    assert awkward1.tolist(one == two) == [False, True, False]
