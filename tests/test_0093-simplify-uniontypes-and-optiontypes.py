# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_numpyarray_merge():
    emptyarray = awkward1.layout.EmptyArray()

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = awkward1.layout.NumpyArray(np1)
    ak2 = awkward1.layout.NumpyArray(np2)

    assert awkward1.to_list(ak1.merge(ak2)) == awkward1.to_list(numpy.concatenate([np1, np2]))
    assert awkward1.to_list(ak1[1:, :-1, ::-1].merge(ak2[1:, :-1, ::-1])) == awkward1.to_list(numpy.concatenate([np1[1:, :-1, ::-1], np2[1:, :-1, ::-1]]))

    for x in [numpy.bool, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64]:
        for y in [numpy.bool, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float32, numpy.float64]:
            z = numpy.concatenate([numpy.array([1, 2, 3], dtype=x), numpy.array([4, 5], dtype=y)]).dtype.type
            one = awkward1.layout.NumpyArray(numpy.array([1, 2, 3], dtype=x))
            two = awkward1.layout.NumpyArray(numpy.array([4, 5], dtype=y))
            three = one.merge(two)
            assert numpy.asarray(three).dtype == numpy.dtype(z), "{0} {1} {2} {3}".format(x, y, z, numpy.asarray(three).dtype.type)
            assert awkward1.to_list(three) == awkward1.to_list(numpy.concatenate([numpy.asarray(one), numpy.asarray(two)]))
            assert awkward1.to_list(one.merge(emptyarray)) == awkward1.to_list(one)
            assert awkward1.to_list(emptyarray.merge(one)) == awkward1.to_list(one)

def test_regulararray_merge():
    emptyarray = awkward1.layout.EmptyArray()

    np1 = numpy.arange(2*7*5).reshape(2, 7, 5)
    np2 = numpy.arange(3*7*5).reshape(3, 7, 5)
    ak1 = awkward1.from_iter(np1, highlevel=False)
    ak2 = awkward1.from_iter(np2, highlevel=False)

    assert awkward1.to_list(ak1.merge(ak2)) == awkward1.to_list(numpy.concatenate([np1, np2]))
    assert awkward1.to_list(ak1.merge(emptyarray)) == awkward1.to_list(ak1)
    assert awkward1.to_list(emptyarray.merge(ak1)) == awkward1.to_list(ak1)

def test_listarray_merge():
    emptyarray = awkward1.layout.EmptyArray()

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
        assert awkward1.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert awkward1.to_list(array2) == [[3, 4, 5, 6], [], [1, 2, 3]]

        assert awkward1.to_list(array1.merge(array2)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [3, 4, 5, 6], [], [1, 2, 3]]
        assert awkward1.to_list(array2.merge(array1)) == [[3, 4, 5, 6], [], [1, 2, 3], [1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert awkward1.to_list(array1.merge(emptyarray)) == awkward1.to_list(array1)
        assert awkward1.to_list(emptyarray.merge(array1)) == awkward1.to_list(array1)

    regulararray = awkward1.layout.RegularArray(content2, 2)
    assert awkward1.to_list(regulararray) == [[1, 2], [3, 4], [5, 6]]
    assert awkward1.to_list(regulararray.merge(emptyarray)) == awkward1.to_list(regulararray)
    assert awkward1.to_list(emptyarray.merge(regulararray)) == awkward1.to_list(regulararray)

    for (dtype1, Index1, ListArray1) in [
            (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32),
            (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32),
            (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64)]:
        starts1 = Index1(numpy.array([0, 3, 3], dtype=dtype1))
        stops1  = Index1(numpy.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert awkward1.to_list(array1.merge(regulararray)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [1, 2], [3, 4], [5, 6]]
        assert awkward1.to_list(regulararray.merge(array1)) == [[1, 2], [3, 4], [5, 6], [1.1, 2.2, 3.3], [], [4.4, 5.5]]

def test_listoffsetarray_merge():
    emptyarray = awkward1.layout.EmptyArray()

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
        assert awkward1.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert awkward1.to_list(array2) == [[2, 3], [], [], [4, 5]]

        assert awkward1.to_list(array1.merge(array2)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [2, 3], [], [], [4, 5]]
        assert awkward1.to_list(array2.merge(array1)) == [[2, 3], [], [], [4, 5], [1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert awkward1.to_list(array1.merge(emptyarray)) == awkward1.to_list(array1)
        assert awkward1.to_list(emptyarray.merge(array1)) == awkward1.to_list(array1)

    regulararray = awkward1.layout.RegularArray(content2, 2)
    assert awkward1.to_list(regulararray) == [[1, 2], [3, 4], [5, 6]]

    for (dtype1, Index1, ListArray1) in [
            (numpy.int32, awkward1.layout.Index32, awkward1.layout.ListArray32),
            (numpy.uint32, awkward1.layout.IndexU32, awkward1.layout.ListArrayU32),
            (numpy.int64, awkward1.layout.Index64, awkward1.layout.ListArray64)]:
        starts1 = Index1(numpy.array([0, 3, 3], dtype=dtype1))
        stops1  = Index1(numpy.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert awkward1.to_list(array1.merge(regulararray)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [1, 2], [3, 4], [5, 6]]
        assert awkward1.to_list(regulararray.merge(array1)) == [[1, 2], [3, 4], [5, 6], [1.1, 2.2, 3.3], [], [4.4, 5.5]]

def test_recordarray_merge():
    emptyarray = awkward1.layout.EmptyArray()

    arrayr1 = awkward1.from_iter([{"x": 0, "y": []}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}], highlevel=False)
    arrayr2 = awkward1.from_iter([{"x": 2.2, "y": [2.2, 2.2]}, {"x": 1.1, "y": [1.1, 1.1]}, {"x": 0.0, "y": [0.0, 0.0]}], highlevel=False)
    arrayr3 = awkward1.from_iter([{"x": 0, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], highlevel=False)
    arrayr4 = awkward1.from_iter([{"y": [], "x": 0}, {"y": [1, 1], "x": 1}, {"y": [2, 2], "x": 2}], highlevel=False)
    arrayr5 = awkward1.from_iter([{"x": 0, "y": [], "z": 0}, {"x": 1, "y": [1, 1], "z": 1}, {"x": 2, "y": [2, 2], "z": 2}], highlevel=False)
    arrayr6 = awkward1.from_iter([{"z": 0, "x": 0, "y": []}, {"z": 1, "x": 1, "y": [1, 1]}, {"z": 2, "x": 2, "y": [2, 2]}], highlevel=False)
    arrayr7 = awkward1.from_iter([{"x": 0}, {"x": 1}, {"x": 2}], highlevel=False)

    arrayt1 = awkward1.from_iter([(0, []), (1, [1.1]), (2, [2, 2])], highlevel=False)
    arrayt2 = awkward1.from_iter([(2.2, [2.2, 2.2]), (1.1, [1.1, 1.1]), (0.0, [0.0, 0.0])], highlevel=False)
    arrayt3 = awkward1.from_iter([(0, 0.0), (1, 1.1), (2, 2.2)], highlevel=False)
    arrayt4 = awkward1.from_iter([([], 0), ([1.1], 1), ([2.2, 2.2], 2)], highlevel=False)
    arrayt5 = awkward1.from_iter([(0, [], 0), (1, [1], 1), (2, [2, 2], 2)], highlevel=False)
    arrayt6 = awkward1.from_iter([(0, 0, []), (1, 1, [1]), (2, 2, [2, 2])], highlevel=False)
    arrayt7 = awkward1.from_iter([(0,), (1,), (2,)], highlevel=False)

    assert arrayr1.mergeable(arrayr2)
    assert arrayr2.mergeable(arrayr1)
    assert not arrayr1.mergeable(arrayr3)
    assert arrayr1.mergeable(arrayr4)
    assert arrayr4.mergeable(arrayr1)
    assert not arrayr1.mergeable(arrayr5)
    assert not arrayr1.mergeable(arrayr6)
    assert arrayr5.mergeable(arrayr6)
    assert arrayr6.mergeable(arrayr5)
    assert not arrayr1.mergeable(arrayr7)

    assert arrayt1.mergeable(arrayt2)
    assert arrayt2.mergeable(arrayt1)
    assert not arrayt1.mergeable(arrayt3)
    assert not arrayt1.mergeable(arrayt4)
    assert not arrayt1.mergeable(arrayt5)
    assert not arrayt1.mergeable(arrayt6)
    assert not arrayt5.mergeable(arrayt6)
    assert not arrayt1.mergeable(arrayt7)

    assert awkward1.to_list(arrayr1.merge(arrayr2)) == [{"x": 0.0, "y": []}, {"x": 1.0, "y": [1.0, 1.0]}, {"x": 2.0, "y": [2.0, 2.0]}, {"x": 2.2, "y": [2.2, 2.2]}, {"x": 1.1, "y": [1.1, 1.1]}, {"x": 0.0, "y": [0.0, 0.0]}]
    assert awkward1.to_list(arrayr2.merge(arrayr1)) == [{"x": 2.2, "y": [2.2, 2.2]}, {"x": 1.1, "y": [1.1, 1.1]}, {"x": 0.0, "y": [0.0, 0.0]}, {"x": 0.0, "y": []}, {"x": 1.0, "y": [1.0, 1.0]}, {"x": 2.0, "y": [2.0, 2.0]}]

    assert awkward1.to_list(arrayr1.merge(arrayr4)) == [{"x": 0, "y": []}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}, {"x": 0, "y": []}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}]
    assert awkward1.to_list(arrayr4.merge(arrayr1)) == [{"x": 0, "y": []}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}, {"x": 0, "y": []}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}]

    assert awkward1.to_list(arrayr5.merge(arrayr6)) == [{"x": 0, "y": [], "z": 0}, {"x": 1, "y": [1, 1], "z": 1}, {"x": 2, "y": [2, 2], "z": 2}, {"x": 0, "y": [], "z": 0}, {"x": 1, "y": [1, 1], "z": 1}, {"x": 2, "y": [2, 2], "z": 2}]
    assert awkward1.to_list(arrayr6.merge(arrayr5)) == [{"x": 0, "y": [], "z": 0}, {"x": 1, "y": [1, 1], "z": 1}, {"x": 2, "y": [2, 2], "z": 2}, {"x": 0, "y": [], "z": 0}, {"x": 1, "y": [1, 1], "z": 1}, {"x": 2, "y": [2, 2], "z": 2}]

    assert awkward1.to_list(arrayt1.merge(arrayt2)) == [(0.0, []), (1.0, [1.1]), (2.0, [2.0, 2.0]), (2.2, [2.2, 2.2]), (1.1, [1.1, 1.1]), (0.0, [0.0, 0.0])]
    assert awkward1.to_list(arrayt2.merge(arrayt1)) == [(2.2, [2.2, 2.2]), (1.1, [1.1, 1.1]), (0.0, [0.0, 0.0]), (0.0, []), (1.0, [1.1]), (2.0, [2.0, 2.0])]

    assert awkward1.to_list(arrayr1.merge(emptyarray)) == awkward1.to_list(arrayr1)
    assert awkward1.to_list(arrayr2.merge(emptyarray)) == awkward1.to_list(arrayr2)
    assert awkward1.to_list(arrayr3.merge(emptyarray)) == awkward1.to_list(arrayr3)
    assert awkward1.to_list(arrayr4.merge(emptyarray)) == awkward1.to_list(arrayr4)
    assert awkward1.to_list(arrayr5.merge(emptyarray)) == awkward1.to_list(arrayr5)
    assert awkward1.to_list(arrayr6.merge(emptyarray)) == awkward1.to_list(arrayr6)
    assert awkward1.to_list(arrayr7.merge(emptyarray)) == awkward1.to_list(arrayr7)

    assert awkward1.to_list(emptyarray.merge(arrayr1)) == awkward1.to_list(arrayr1)
    assert awkward1.to_list(emptyarray.merge(arrayr2)) == awkward1.to_list(arrayr2)
    assert awkward1.to_list(emptyarray.merge(arrayr3)) == awkward1.to_list(arrayr3)
    assert awkward1.to_list(emptyarray.merge(arrayr4)) == awkward1.to_list(arrayr4)
    assert awkward1.to_list(emptyarray.merge(arrayr5)) == awkward1.to_list(arrayr5)
    assert awkward1.to_list(emptyarray.merge(arrayr6)) == awkward1.to_list(arrayr6)
    assert awkward1.to_list(emptyarray.merge(arrayr7)) == awkward1.to_list(arrayr7)

    assert awkward1.to_list(arrayt1.merge(emptyarray)) == awkward1.to_list(arrayt1)
    assert awkward1.to_list(arrayt2.merge(emptyarray)) == awkward1.to_list(arrayt2)
    assert awkward1.to_list(arrayt3.merge(emptyarray)) == awkward1.to_list(arrayt3)
    assert awkward1.to_list(arrayt4.merge(emptyarray)) == awkward1.to_list(arrayt4)
    assert awkward1.to_list(arrayt5.merge(emptyarray)) == awkward1.to_list(arrayt5)
    assert awkward1.to_list(arrayt6.merge(emptyarray)) == awkward1.to_list(arrayt6)
    assert awkward1.to_list(arrayt7.merge(emptyarray)) == awkward1.to_list(arrayt7)

    assert awkward1.to_list(emptyarray.merge(arrayt1)) == awkward1.to_list(arrayt1)
    assert awkward1.to_list(emptyarray.merge(arrayt2)) == awkward1.to_list(arrayt2)
    assert awkward1.to_list(emptyarray.merge(arrayt3)) == awkward1.to_list(arrayt3)
    assert awkward1.to_list(emptyarray.merge(arrayt4)) == awkward1.to_list(arrayt4)
    assert awkward1.to_list(emptyarray.merge(arrayt5)) == awkward1.to_list(arrayt5)
    assert awkward1.to_list(emptyarray.merge(arrayt6)) == awkward1.to_list(arrayt6)
    assert awkward1.to_list(emptyarray.merge(arrayt7)) == awkward1.to_list(arrayt7)

def test_indexedarray_merge():
    emptyarray = awkward1.layout.EmptyArray()

    content1 = awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content2 = awkward1.from_iter([[1, 2], [], [3, 4]], highlevel=False)
    index1 = awkward1.layout.Index64(numpy.array([2, 0, -1, 0, 1, 2], dtype=numpy.int64))
    indexedarray1 = awkward1.layout.IndexedOptionArray64(index1, content1)
    assert awkward1.to_list(indexedarray1) == [[4.4, 5.5], [1.1, 2.2, 3.3], None, [1.1, 2.2, 3.3], [], [4.4, 5.5]]

    assert awkward1.to_list(indexedarray1.merge(content2)) == [[4.4, 5.5], [1.1, 2.2, 3.3], None, [1.1, 2.2, 3.3], [], [4.4, 5.5], [1.0, 2.0], [], [3.0, 4.0]]
    assert awkward1.to_list(content2.merge(indexedarray1)) == [[1.0, 2.0], [], [3.0, 4.0], [4.4, 5.5], [1.1, 2.2, 3.3], None, [1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(indexedarray1.merge(indexedarray1)) == [[4.4, 5.5], [1.1, 2.2, 3.3], None, [1.1, 2.2, 3.3], [], [4.4, 5.5], [4.4, 5.5], [1.1, 2.2, 3.3], None, [1.1, 2.2, 3.3], [], [4.4, 5.5]]

def test_unionarray_merge():
    emptyarray = awkward1.layout.EmptyArray()

    one = awkward1.from_iter([0.0, 1.1, 2.2, [], [1], [2, 2]], highlevel=False)
    two = awkward1.from_iter([{"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}], highlevel=False)
    three = awkward1.from_iter(["one", "two", "three"], highlevel=False)

    assert awkward1.to_list(one.merge(two)) == [0.0, 1.1, 2.2, [], [1], [2, 2], {"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}]
    assert awkward1.to_list(two.merge(one)) == [{"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}, 0.0, 1.1, 2.2, [], [1], [2, 2]]

    assert awkward1.to_list(one.merge(emptyarray)) == [0.0, 1.1, 2.2, [], [1], [2, 2]]
    assert awkward1.to_list(emptyarray.merge(one)) == [0.0, 1.1, 2.2, [], [1], [2, 2]]

    assert awkward1.to_list(one.merge(three)) == [0.0, 1.1, 2.2, [], [1], [2, 2], "one", "two", "three"]
    assert awkward1.to_list(two.merge(three)) == [{"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}, "one", "two", "three"]
    assert awkward1.to_list(three.merge(one)) == ["one", "two", "three", 0.0, 1.1, 2.2, [], [1], [2, 2]]
    assert awkward1.to_list(three.merge(two)) == ["one", "two", "three", {"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}]

def test_merge_parameters():
    one = awkward1.from_iter([[121, 117, 99, 107, 121], [115, 116, 117, 102, 102]], highlevel=False)
    two = awkward1.from_iter(["good", "stuff"], highlevel=False)

    assert awkward1.to_list(one.merge(two)) == [[121, 117, 99, 107, 121], [115, 116, 117, 102, 102], "good", "stuff"]
    assert awkward1.to_list(two.merge(one)) == ["good", "stuff", [121, 117, 99, 107, 121], [115, 116, 117, 102, 102]]

def test_bytemask():
    array = awkward1.from_iter(["one", "two", None, "three", None, None, "four"], highlevel=False)
    assert numpy.asarray(array.bytemask()).tolist() == [0, 0, 1, 0, 1, 1, 0]

    index2 = awkward1.layout.Index64(numpy.array([2, 2, 1, 5, 0], dtype=numpy.int64))
    array2 = awkward1.layout.IndexedArray64(index2, array)
    assert numpy.asarray(array2.bytemask()).tolist() == [0, 0, 0, 0, 0]

def test_indexedarray_simplify():
    array = awkward1.from_iter(["one", "two", None, "three", None, None, "four", "five"], highlevel=False)
    assert numpy.asarray(array.index).tolist() == [0, 1, -1, 2, -1, -1, 3, 4]

    index2 = awkward1.layout.Index64(numpy.array([2, 2, 1, 6, 5], dtype=numpy.int64))
    array2 = awkward1.layout.IndexedArray64(index2, array)
    assert awkward1.to_list(array2.simplify()) == awkward1.to_list(array2) == [None, None, "two", "four", None]

def test_indexedarray_simplify_more():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))

    index1_32 = awkward1.layout.Index32(numpy.array([6, 5, 4, 3, 2, 1, 0], dtype=numpy.int32))
    index1_U32 = awkward1.layout.IndexU32(numpy.array([6, 5, 4, 3, 2, 1, 0], dtype=numpy.uint32))
    index1_64 = awkward1.layout.Index64(numpy.array([6, 5, 4, 3, 2, 1, 0], dtype=numpy.int64))
    index2_32 = awkward1.layout.Index32(numpy.array([0, 2, 4, 6], dtype=numpy.int32))
    index2_U32 = awkward1.layout.IndexU32(numpy.array([0, 2, 4, 6], dtype=numpy.uint32))
    index2_64 = awkward1.layout.Index64(numpy.array([0, 2, 4, 6], dtype=numpy.int64))

    array = awkward1.layout.IndexedArray32(index2_32, awkward1.layout.IndexedArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray32(index2_32, awkward1.layout.IndexedArrayU32(index1_U32, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray32(index2_32, awkward1.layout.IndexedArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArrayU32(index2_U32, awkward1.layout.IndexedArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArrayU32(index2_U32, awkward1.layout.IndexedArrayU32(index1_U32, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArrayU32(index2_U32, awkward1.layout.IndexedArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray64(index2_64, awkward1.layout.IndexedArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray64(index2_64, awkward1.layout.IndexedArrayU32(index1_U32, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray64(index2_64, awkward1.layout.IndexedArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    index1_32 = awkward1.layout.Index32(numpy.array([6, 5, -1, 3, -1, 1, 0], dtype=numpy.int32))
    index1_64 = awkward1.layout.Index64(numpy.array([6, 5, -1, 3, -1, 1, 0], dtype=numpy.int64))
    index2_32 = awkward1.layout.Index32(numpy.array([0, 2, 4, 6], dtype=numpy.int32))
    index2_U32 = awkward1.layout.IndexU32(numpy.array([0, 2, 4, 6], dtype=numpy.uint32))
    index2_64 = awkward1.layout.Index64(numpy.array([0, 2, 4, 6], dtype=numpy.int64))

    array = awkward1.layout.IndexedArray32(index2_32, awkward1.layout.IndexedOptionArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, None, None, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray32(index2_32, awkward1.layout.IndexedOptionArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, None, None, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArrayU32(index2_U32, awkward1.layout.IndexedOptionArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, None, None, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArrayU32(index2_U32, awkward1.layout.IndexedOptionArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, None, None, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray64(index2_64, awkward1.layout.IndexedOptionArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, None, None, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedArray64(index2_64, awkward1.layout.IndexedOptionArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, None, None, 0.0]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    index1_32 = awkward1.layout.Index32(numpy.array([6, 5, 4, 3, 2, 1, 0], dtype=numpy.int32))
    index1_U32 = awkward1.layout.IndexU32(numpy.array([6, 5, 4, 3, 2, 1, 0], dtype=numpy.uint32))
    index1_64 = awkward1.layout.Index64(numpy.array([6, 5, 4, 3, 2, 1, 0], dtype=numpy.int64))
    index2_32 = awkward1.layout.Index32(numpy.array([0, -1, 4, -1], dtype=numpy.int32))
    index2_64 = awkward1.layout.Index64(numpy.array([0, -1, 4, -1], dtype=numpy.int64))

    array = awkward1.layout.IndexedOptionArray32(index2_32, awkward1.layout.IndexedArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, None, 2.2, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray32(index2_32, awkward1.layout.IndexedArrayU32(index1_U32, content))
    assert awkward1.to_list(array) == [6.6, None, 2.2, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray32(index2_32, awkward1.layout.IndexedArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, None, 2.2, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray64(index2_64, awkward1.layout.IndexedArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, None, 2.2, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray64(index2_64, awkward1.layout.IndexedArrayU32(index1_U32, content))
    assert awkward1.to_list(array) == [6.6, None, 2.2, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray64(index2_64, awkward1.layout.IndexedArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, None, 2.2, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    index1_32 = awkward1.layout.Index32(numpy.array([6, 5, -1, 3, -1, 1, 0], dtype=numpy.int32))
    index1_64 = awkward1.layout.Index64(numpy.array([6, 5, -1, 3, -1, 1, 0], dtype=numpy.int64))
    index2_32 = awkward1.layout.Index32(numpy.array([0, -1, 4, -1], dtype=numpy.int32))
    index2_64 = awkward1.layout.Index64(numpy.array([0, -1, 4, -1], dtype=numpy.int64))

    array = awkward1.layout.IndexedOptionArray32(index2_32, awkward1.layout.IndexedOptionArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, None, None, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray32(index2_32, awkward1.layout.IndexedOptionArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, None, None, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray64(index2_64, awkward1.layout.IndexedOptionArray32(index1_32, content))
    assert awkward1.to_list(array) == [6.6, None, None, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

    array = awkward1.layout.IndexedOptionArray64(index2_64, awkward1.layout.IndexedOptionArray64(index1_64, content))
    assert awkward1.to_list(array) == [6.6, None, None, None]
    assert awkward1.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), awkward1.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, awkward1.layout.NumpyArray)

def test_unionarray_simplify_one():
    one = awkward1.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = awkward1.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = awkward1.from_iter([1.1, 2.2, 3.3], highlevel=False)
    tags  =  awkward1.layout.Index8(numpy.array([0, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 0], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 1, 0, 0, 1, 2, 1, 2, 3, 3, 2, 4], dtype=numpy.int64))
    array = awkward1.layout.UnionArray8_64(tags, index, [one, two, three])

    assert awkward1.to_list(array) == [5, 4, [], 1.1, [1], 3, 2.2, [2, 2], [3, 3, 3], 2, 3.3, 1]
    assert awkward1.to_list(array.simplify()) == [5.0, 4.0, [], 1.1, [1], 3.0, 2.2, [2, 2], [3, 3, 3], 2.0, 3.3, 1.0]
    assert len(array.contents) == 3
    assert len(array.simplify().contents) == 2

def test_unionarray_simplify():
    one = awkward1.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = awkward1.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = awkward1.from_iter([1.1, 2.2, 3.3], highlevel=False)

    tags2  =  awkward1.layout.Index8(numpy.array([0, 1, 0, 1, 0, 0, 1], dtype=numpy.int8))
    index2 = awkward1.layout.Index32(numpy.array([0, 0, 1, 1, 2, 3, 2], dtype=numpy.int32))
    inner = awkward1.layout.UnionArray8_32(tags2, index2, [two, three])
    tags1  =  awkward1.layout.Index8(numpy.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=numpy.int8))
    index1 = awkward1.layout.Index64(numpy.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=numpy.int64))
    outer = awkward1.layout.UnionArray8_64(tags1, index1, [one, inner])
    assert awkward1.to_list(outer) == [5, 4, [], 1.1, [1], 3, 2.2, [2, 2], [3, 3, 3], 2, 3.3, 1]

    assert awkward1.to_list(outer.simplify()) == [5.0, 4.0, [], 1.1, [1], 3.0, 2.2, [2, 2], [3, 3, 3], 2.0, 3.3, 1.0]
    assert isinstance(outer.content(1), awkward1.layout.UnionArray8_32)
    assert isinstance(outer.simplify().content(0), awkward1.layout.NumpyArray)
    assert isinstance(outer.simplify().content(1), awkward1.layout.ListOffsetArray64)
    assert len(outer.simplify().contents) == 2

    tags2  =  awkward1.layout.Index8(numpy.array([0, 1, 0, 1, 0, 0, 1], dtype=numpy.int8))
    index2 = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 3, 2], dtype=numpy.int64))
    inner = awkward1.layout.UnionArray8_64(tags2, index2, [two, three])
    tags1  =  awkward1.layout.Index8(numpy.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=numpy.int8))
    index1 = awkward1.layout.Index32(numpy.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=numpy.int32))
    outer = awkward1.layout.UnionArray8_32(tags1, index1, [inner, one])
    assert awkward1.to_list(outer) == [5, 4, [], 1.1, [1], 3, 2.2, [2, 2], [3, 3, 3], 2, 3.3, 1]

def test_concatenate():
    one = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)
    two = awkward1.Array([[], [1], [2, 2], [3, 3, 3]], check_valid=True)
    three = awkward1.Array([True, False, False, True, True], check_valid=True)

    assert awkward1.to_list(awkward1.concatenate([one, two, three])) == [1.1, 2.2, 3.3, 4.4, 5.5, [], [1], [2, 2], [3, 3, 3], 1.0, 0.0, 0.0, 1.0, 1.0]
    assert isinstance(awkward1.concatenate([one, two, three], highlevel=False), awkward1.layout.UnionArray8_64)
    assert len(awkward1.concatenate([one, two, three], highlevel=False).contents) == 2

def test_where():
    condition = awkward1.Array([True, False, True, False, True], check_valid=True)
    one = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)
    two = awkward1.Array([False, False, False, True, True], check_valid=True)
    three = awkward1.Array([[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4]], check_valid=True)

    assert awkward1.to_list(awkward1.where(condition, one, two)) == [1.1, 0.0, 3.3, 1.0, 5.5]
    assert awkward1.to_list(awkward1.where(condition, one, three)) == [1.1, [1], 3.3, [3, 3, 3], 5.5]
