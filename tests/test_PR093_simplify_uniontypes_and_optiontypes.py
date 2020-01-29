# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_numpyarray_merge():
    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = awkward1.layout.NumpyArray(np1)
    ak2 = awkward1.layout.NumpyArray(np2)

    assert awkward1.tolist(ak1.merge(ak2)) == awkward1.tolist(numpy.concatenate([np1, np2]))
    assert awkward1.tolist(ak1[1:, :-1, ::-1].merge(ak2[1:, :-1, ::-1])) == awkward1.tolist(numpy.concatenate([np1[1:, :-1, ::-1], np2[1:, :-1, ::-1]]))

    for x, y, z in [(numpy.double, numpy.double, numpy.double),
                    (numpy.double, numpy.float,  numpy.double),
                    (numpy.double, numpy.int64,  numpy.double),
                    (numpy.double, numpy.uint64, numpy.double),
                    (numpy.double, numpy.int32,  numpy.double),
                    (numpy.double, numpy.uint32, numpy.double),
                    (numpy.double, numpy.int16,  numpy.double),
                    (numpy.double, numpy.uint16, numpy.double),
                    (numpy.double, numpy.int8,   numpy.double),
                    (numpy.double, numpy.uint8,  numpy.double),
                    (numpy.double, numpy.bool,   numpy.double),
                    (numpy.float,  numpy.double, numpy.double),
                    (numpy.float,  numpy.float,  numpy.double),
                    (numpy.float,  numpy.int64,  numpy.double),
                    (numpy.float,  numpy.uint64, numpy.double),
                    (numpy.float,  numpy.int32,  numpy.double),
                    (numpy.float,  numpy.uint32, numpy.double),
                    (numpy.float,  numpy.int16,  numpy.double),
                    (numpy.float,  numpy.uint16, numpy.double),
                    (numpy.float,  numpy.int8,   numpy.double),
                    (numpy.float,  numpy.uint8,  numpy.double),
                    (numpy.float,  numpy.bool,   numpy.double),
                    (numpy.int64,  numpy.double, numpy.double),
                    (numpy.int64,  numpy.float,  numpy.double),
                    (numpy.int64,  numpy.int64,  numpy.int64),
                    (numpy.int64,  numpy.uint64, numpy.int64),
                    (numpy.int64,  numpy.int32,  numpy.int64),
                    (numpy.int64,  numpy.uint32, numpy.int64),
                    (numpy.int64,  numpy.int16,  numpy.int64),
                    (numpy.int64,  numpy.uint16, numpy.int64),
                    (numpy.int64,  numpy.int8,   numpy.int64),
                    (numpy.int64,  numpy.uint8,  numpy.int64),
                    (numpy.int64,  numpy.bool,   numpy.int64),
                    (numpy.uint64, numpy.double, numpy.double),
                    (numpy.uint64, numpy.float,  numpy.double),
                    (numpy.uint64, numpy.int64,  numpy.int64),
                    (numpy.uint64, numpy.uint64, numpy.uint64),
                    (numpy.uint64, numpy.int32,  numpy.int64),
                    (numpy.uint64, numpy.uint32, numpy.int64),
                    (numpy.uint64, numpy.int16,  numpy.int64),
                    (numpy.uint64, numpy.uint16, numpy.int64),
                    (numpy.uint64, numpy.int8,   numpy.int64),
                    (numpy.uint64, numpy.uint8,  numpy.int64),
                    (numpy.uint64, numpy.bool,   numpy.int64),
                    (numpy.int32,  numpy.double, numpy.double),
                    (numpy.int32,  numpy.float,  numpy.double),
                    (numpy.int32,  numpy.int64,  numpy.int64),
                    (numpy.int32,  numpy.uint64, numpy.int64),
                    (numpy.int32,  numpy.int32,  numpy.int64),
                    (numpy.int32,  numpy.uint32, numpy.int64),
                    (numpy.int32,  numpy.int16,  numpy.int64),
                    (numpy.int32,  numpy.uint16, numpy.int64),
                    (numpy.int32,  numpy.int8,   numpy.int64),
                    (numpy.int32,  numpy.uint8,  numpy.int64),
                    (numpy.int32,  numpy.bool,   numpy.int64),
                    (numpy.uint32, numpy.double, numpy.double),
                    (numpy.uint32, numpy.float,  numpy.double),
                    (numpy.uint32, numpy.int64,  numpy.int64),
                    (numpy.uint32, numpy.uint64, numpy.int64),
                    (numpy.uint32, numpy.int32,  numpy.int64),
                    (numpy.uint32, numpy.uint32, numpy.int64),
                    (numpy.uint32, numpy.int16,  numpy.int64),
                    (numpy.uint32, numpy.uint16, numpy.int64),
                    (numpy.uint32, numpy.int8,   numpy.int64),
                    (numpy.uint32, numpy.uint8,  numpy.int64),
                    (numpy.uint32, numpy.bool,   numpy.int64),
                    (numpy.int16,  numpy.double, numpy.double),
                    (numpy.int16,  numpy.float,  numpy.double),
                    (numpy.int16,  numpy.int64,  numpy.int64),
                    (numpy.int16,  numpy.uint64, numpy.int64),
                    (numpy.int16,  numpy.int32,  numpy.int64),
                    (numpy.int16,  numpy.uint32, numpy.int64),
                    (numpy.int16,  numpy.int16,  numpy.int64),
                    (numpy.int16,  numpy.uint16, numpy.int64),
                    (numpy.int16,  numpy.int8,   numpy.int64),
                    (numpy.int16,  numpy.uint8,  numpy.int64),
                    (numpy.int16,  numpy.bool,   numpy.int64),
                    (numpy.uint16, numpy.double, numpy.double),
                    (numpy.uint16, numpy.float,  numpy.double),
                    (numpy.uint16, numpy.int64,  numpy.int64),
                    (numpy.uint16, numpy.uint64, numpy.int64),
                    (numpy.uint16, numpy.int32,  numpy.int64),
                    (numpy.uint16, numpy.uint32, numpy.int64),
                    (numpy.uint16, numpy.int16,  numpy.int64),
                    (numpy.uint16, numpy.uint16, numpy.int64),
                    (numpy.uint16, numpy.int8,   numpy.int64),
                    (numpy.uint16, numpy.uint8,  numpy.int64),
                    (numpy.uint16, numpy.bool,   numpy.int64),
                    (numpy.int8,   numpy.double, numpy.double),
                    (numpy.int8,   numpy.float,  numpy.double),
                    (numpy.int8,   numpy.int64,  numpy.int64),
                    (numpy.int8,   numpy.uint64, numpy.int64),
                    (numpy.int8,   numpy.int32,  numpy.int64),
                    (numpy.int8,   numpy.uint32, numpy.int64),
                    (numpy.int8,   numpy.int16,  numpy.int64),
                    (numpy.int8,   numpy.uint16, numpy.int64),
                    (numpy.int8,   numpy.int8,   numpy.int64),
                    (numpy.int8,   numpy.uint8,  numpy.int64),
                    (numpy.int8,   numpy.bool,   numpy.int64),
                    (numpy.uint8,  numpy.double, numpy.double),
                    (numpy.uint8,  numpy.float,  numpy.double),
                    (numpy.uint8,  numpy.int64,  numpy.int64),
                    (numpy.uint8,  numpy.uint64, numpy.int64),
                    (numpy.uint8,  numpy.int32,  numpy.int64),
                    (numpy.uint8,  numpy.uint32, numpy.int64),
                    (numpy.uint8,  numpy.int16,  numpy.int64),
                    (numpy.uint8,  numpy.uint16, numpy.int64),
                    (numpy.uint8,  numpy.int8,   numpy.int64),
                    (numpy.uint8,  numpy.uint8,  numpy.int64),
                    (numpy.uint8,  numpy.bool,   numpy.int64),
                    (numpy.bool,   numpy.double, numpy.double),
                    (numpy.bool,   numpy.float,  numpy.double),
                    (numpy.bool,   numpy.int64,  numpy.int64),
                    (numpy.bool,   numpy.uint64, numpy.int64),
                    (numpy.bool,   numpy.int32,  numpy.int64),
                    (numpy.bool,   numpy.uint32, numpy.int64),
                    (numpy.bool,   numpy.int16,  numpy.int64),
                    (numpy.bool,   numpy.uint16, numpy.int64),
                    (numpy.bool,   numpy.int8,   numpy.int64),
                    (numpy.bool,   numpy.uint8,  numpy.int64),
                    (numpy.bool,   numpy.bool,   numpy.bool)]:
        one = awkward1.layout.NumpyArray(numpy.array([1, 2, 3], dtype=x))
        two = awkward1.layout.NumpyArray(numpy.array([4, 5], dtype=y))
        three = one.merge(two)
        assert numpy.asarray(three).dtype == numpy.dtype(z)
        assert awkward1.tolist(three) == awkward1.tolist(numpy.concatenate([numpy.asarray(one), numpy.asarray(two)]))

def test_regulararray_merge():
    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = awkward1.Array(np1).layout
    ak2 = awkward1.Array(np2).layout
    assert awkward1.tolist(ak1.merge(ak2)) == awkward1.tolist(numpy.concatenate([np1, np2]))

def test_listarray_merge():
    content1 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7]))

    for (dtype1, Index1, ListArray1), (dtype2, Index2, ListArray2) in [
            ((numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32),    (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32)),
            ((numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32),    (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32)),
            ((numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32),    (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64)),
            ((numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32), (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32)),
            ((numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32), (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32)),
            ((numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32), (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64)),
            ((numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64),    (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32)),
            ((numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64),    (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32)),
            ((numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64),    (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64))]:
        starts1 = Index1(numpy.array([0, 3, 3], dtype=dtype1))
        stops1  = Index1(numpy.array([3, 3, 5], dtype=dtype1))
        starts2 = Index2(numpy.array([2, 99, 0], dtype=dtype2))
        stops2  = Index2(numpy.array([6, 99, 3], dtype=dtype2))
        array1 = ListArray1(starts1, stops1, content1)
        array2 = ListArray2(starts2, stops2, content2)
        assert awkward1.tolist(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert awkward1.tolist(array2) == [[3, 4, 5, 6], [], [1, 2, 3]]

        assert awkward1.tolist(array1.merge(array2)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [3, 4, 5, 6], [], [1, 2, 3]]
        assert awkward1.tolist(array2.merge(array1)) == [[3, 4, 5, 6], [], [1, 2, 3], [1.1, 2.2, 3.3], [], [4.4, 5.5]]

    regulararray = awkward1.layout.RegularArray(content2, 2)
    assert awkward1.tolist(regulararray) == [[1, 2], [3, 4], [5, 6]]

    for (dtype1, Index1, ListArray1) in [
            (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32),
            (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32),
            (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64)]:
        starts1 = Index1(numpy.array([0, 3, 3], dtype=dtype1))
        stops1  = Index1(numpy.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert awkward1.tolist(array1.merge(regulararray)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [1, 2], [3, 4], [5, 6]]
        assert awkward1.tolist(regulararray.merge(array1)) == [[1, 2], [3, 4], [5, 6], [1.1, 2.2, 3.3], [], [4.4, 5.5]]

def test_listoffsetarray_merge():
    content1 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7]))

    for (dtype1, Index1, ListOffsetArray1), (dtype2, Index2, ListOffsetArray2) in [
            ((numpy.int32, awkward1.layout.Index32, awkward1.layout.ListOffsetArray32),    (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListOffsetArray32)),
            ((numpy.int32, awkward1.layout.Index32, awkward1.layout.ListOffsetArray32),    (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListOffsetArrayU32)),
            ((numpy.int32, awkward1.layout.Index32, awkward1.layout.ListOffsetArray32),    (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListOffsetArray64)),
            ((numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListOffsetArrayU32), (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListOffsetArray32)),
            ((numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListOffsetArrayU32), (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListOffsetArrayU32)),
            ((numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListOffsetArrayU32), (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListOffsetArray64)),
            ((numpy.int64, awkward1.layout.Index64, awkward1.layout.ListOffsetArray64),    (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListOffsetArray32)),
            ((numpy.int64, awkward1.layout.Index64, awkward1.layout.ListOffsetArray64),    (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListOffsetArrayU32)),
            ((numpy.int64, awkward1.layout.Index64, awkward1.layout.ListOffsetArray64),    (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListOffsetArray64))]:
        offsets1 = Index1(numpy.array([0, 3, 3, 5], dtype=dtype1))
        offsets2 = Index2(numpy.array([1, 3, 3, 3, 5], dtype=dtype2))
        array1 = ListOffsetArray1(offsets1, content1)
        array2 = ListOffsetArray2(offsets2, content2)
        assert awkward1.tolist(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert awkward1.tolist(array2) == [[2, 3], [], [], [4, 5]]

        assert awkward1.tolist(array1.merge(array2)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [2, 3], [], [], [4, 5]]
        assert awkward1.tolist(array2.merge(array1)) == [[2, 3], [], [], [4, 5], [1.1, 2.2, 3.3], [], [4.4, 5.5]]

    regulararray = awkward1.layout.RegularArray(content2, 2)
    assert awkward1.tolist(regulararray) == [[1, 2], [3, 4], [5, 6]]

    for (dtype1, Index1, ListArray1) in [
            (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32),
            (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32),
            (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64)]:
        starts1 = Index1(numpy.array([0, 3, 3], dtype=dtype1))
        stops1  = Index1(numpy.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert awkward1.tolist(array1.merge(regulararray)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [1, 2], [3, 4], [5, 6]]
        assert awkward1.tolist(regulararray.merge(array1)) == [[1, 2], [3, 4], [5, 6], [1.1, 2.2, 3.3], [], [4.4, 5.5]]
