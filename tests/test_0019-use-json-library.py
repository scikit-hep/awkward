# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import os
import json

import pytest
import numpy

import awkward1

def test_fromstring():
    a = awkward1.from_json("[[1.1, 2.2, 3], [], [4, 5.5]]")
    assert awkward1.to_list(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        awkward1.from_json("[[1.1, 2.2, 3], [blah], [4, 5.5]]")

def test_fromfile(tmp_path):
    with open(os.path.join(str(tmp_path), "tmp1.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], [], [4, 5.5]]")

    a = awkward1.from_json(os.path.join(str(tmp_path), "tmp1.json"))
    assert awkward1.to_list(a) == [[1.1, 2.2, 3.0], [], [4.0, 5.5]]

    with pytest.raises(ValueError):
        awkward1.from_json("nonexistent.json")

    with open(os.path.join(str(tmp_path), "tmp2.json"), "w") as f:
        f.write("[[1.1, 2.2, 3], []], [4, 5.5]]")

    with pytest.raises(ValueError):
        awkward1.from_json(os.path.join(str(tmp_path), "tmp2.json"))

def test_tostring():
    content = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(-1, 7))
    offsetsA = numpy.arange(0, 2*3*5 + 5, 5)
    offsetsB = numpy.arange(0, 2*3 + 3, 3)
    startsA, stopsA = offsetsA[:-1], offsetsA[1:]
    startsB, stopsB = offsetsB[:-1], offsetsB[1:]

    listoffsetarrayA32 = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(offsetsA), content)
    listarrayA32 = awkward1.layout.ListArray32(awkward1.layout.Index32(startsA), awkward1.layout.Index32(stopsA), content)
    modelA = numpy.arange(2*3*5*7).reshape(2*3, 5, 7)

    listoffsetarrayB32 = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(offsetsB), listoffsetarrayA32)
    listarrayB32 = awkward1.layout.ListArray32(awkward1.layout.Index32(startsB), awkward1.layout.Index32(stopsB), listarrayA32)
    modelB = numpy.arange(2*3*5*7).reshape(2, 3, 5, 7)

    assert content.tojson() == json.dumps(awkward1.to_list(content), separators=(",", ":"))
    assert listoffsetarrayA32.tojson() == json.dumps(modelA.tolist(), separators=(",", ":"))
    assert listoffsetarrayB32.tojson() == json.dumps(modelB.tolist(), separators=(",", ":"))
    awkward1.to_json(awkward1.from_json("[[1.1,2.2,3],[],[4,5.5]]")) == "[[1.1,2.2,3],[],[4,5.5]]"

def test_tofile(tmp_path):
    awkward1.to_json(awkward1.from_json("[[1.1,2.2,3],[],[4,5.5]]"), os.path.join(str(tmp_path), "tmp1.json"))

    with open(os.path.join(str(tmp_path), "tmp1.json"), "r") as f:
        f.read() == "[[1.1,2.2,3],[],[4,5.5]]"

def test_root_nestedvector():
    # fromcounts([3, 2], fromcounts([1, 0, 2, 2, 1], [123, 99, 123, 99, 123, 123]))
    # <JaggedArray [[[123] [] [99 123]] [[99 123] [123]]]>

    # outer offsets: [0, 3, 5]
    # inner offsets: [0, 1, 1, 3, 5, 6]

    byteoffsets = awkward1.layout.Index64(numpy.array([0, 28, 52], dtype=numpy.int64))
    rawdata = awkward1.layout.NumpyArray(numpy.array([
        0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 99, 0, 0, 0, 123,
        0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 99, 0, 0, 0, 123, 0, 0, 0, 1, 0, 0, 0, 123
        ], dtype=numpy.uint8))

    result = awkward1._ext.fromroot_nestedvector(byteoffsets, rawdata, 2, numpy.dtype(">i").itemsize, ">i")
    assert numpy.asarray(result.offsets).tolist() == [0, 3, 5]
    assert numpy.asarray(result.content.offsets).tolist() == [0, 1, 1, 3, 5, 6]
    assert numpy.asarray(result.content.content).tolist() == [123, 99, 123, 99, 123, 123]
    assert awkward1.to_list(result) == [[[123], [], [99, 123]], [[99, 123], [123]]]

def test_fromiter():
    assert awkward1.to_list(awkward1.from_iter([True, True, False, False, True])) == [True, True, False, False, True]
    assert awkward1.to_list(awkward1.from_iter([5, 4, 3, 2, 1])) == [5, 4, 3, 2, 1]
    assert awkward1.to_list(awkward1.from_iter([5, 4, 3.14, 2.22, 1.23])) == [5.0, 4.0, 3.14, 2.22, 1.23]
    assert awkward1.to_list(awkward1.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(awkward1.from_iter([[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]])) == [[[1.1, 2.2, 3.3], []], [[4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]

def test_numpy():
    a = awkward1.layout.NumpyArray(numpy.arange(2*3*5).reshape(2, 3, 5))
    assert awkward1.to_list(a) == [[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]]
    assert awkward1.to_json(a) == "[[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]],[[15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29]]]"

    b = awkward1.layout.NumpyArray(numpy.array([[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], [[10.1, 20.2, 30.3], [40.4, 50.5, 60.6]]]))
    assert awkward1.to_json(b) == "[[[1.1,2.2,3.3],[4.4,5.5,6.6]],[[10.1,20.2,30.3],[40.4,50.5,60.6]]]"

    c = awkward1.layout.NumpyArray(numpy.array([[True, False, True], [False, False, True]]))
    assert awkward1.to_json(c) == "[[true,false,true],[false,false,true]]"
