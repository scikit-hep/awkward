# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_basic():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray()
    recordarray.append(content1, "one")
    recordarray.append(listoffsetarray, "two")
    recordarray.append(content2)
    recordarray.setkey(0, "wonky")
    assert awkward1.tolist(recordarray.field(0)) == [1, 2, 3, 4, 5]
    assert awkward1.tolist(recordarray.field("two")) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(recordarray.field("wonky")) == [1, 2, 3, 4, 5]

    str(recordarray)
    assert awkward1.tojson(recordarray) == '[{"wonky":1,"two":[1.1,2.2,3.3],"2":1.1},{"wonky":2,"two":[],"2":2.2},{"wonky":3,"two":[4.4,5.5],"2":3.3},{"wonky":4,"two":[6.6],"2":4.4},{"wonky":5,"two":[7.7,8.8,9.9],"2":5.5}]'

    assert len(recordarray) == 5
    assert recordarray.key(0) == "wonky"
    assert recordarray.key(1) == "two"
    assert recordarray.key(2) == "2"
    assert recordarray.index("wonky") == 0
    assert recordarray.index("one") == 0
    assert recordarray.index("0") == 0
    assert recordarray.index("two") == 1
    assert recordarray.index("1") == 1
    assert recordarray.index("2") == 2
    assert recordarray.has("wonky")
    assert recordarray.has("one")
    assert recordarray.has("0")
    assert recordarray.has("two")
    assert recordarray.has("1")
    assert recordarray.has("2")
    assert set(recordarray.aliases(0)) == set(["wonky", "one", "0"])
    assert set(recordarray.aliases("wonky")) == set(["wonky", "one", "0"])
    assert set(recordarray.aliases("one")) == set(["wonky", "one", "0"])
    assert set(recordarray.aliases("0")) == set(["wonky", "one", "0"])
    assert set(recordarray.aliases(1)) == set(["two", "1"])
    assert set(recordarray.aliases("two")) == set(["two", "1"])
    assert set(recordarray.aliases("1")) == set(["two", "1"])
    assert set(recordarray.aliases(2)) == set(["2"])
    assert set(recordarray.aliases("2")) == set(["2"])

def test_scalar_record():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray()
    recordarray.append(content1, "one")
    recordarray.append(listoffsetarray, "two")

    str(recordarray)
    str(recordarray[2])
