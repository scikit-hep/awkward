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

    assert recordarray.keys() == ["wonky", "two", "2"]
    assert [awkward1.tolist(x) for x in recordarray.values()] == [[1, 2, 3, 4, 5], [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]]
    pairs = recordarray.items()
    assert pairs[0][0] == "wonky"
    assert pairs[1][0] == "two"
    assert pairs[2][0] == "2"
    assert awkward1.tolist(pairs[0][1]) == [1, 2, 3, 4, 5]
    assert awkward1.tolist(pairs[1][1]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(pairs[2][1]) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

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
    assert awkward1.tojson(recordarray[2]) == '{"one":3,"two":[4.4,5.5]}'

    assert recordarray[2].keys() == ["one", "two"]
    assert [awkward1.tolist(x) for x in recordarray[2].values()] == [3, [4.4, 5.5]]
    pairs = recordarray[2].items()
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[0][1] == 3
    assert awkward1.tolist(pairs[1][1]) == [4.4, 5.5]
    assert awkward1.tolist(recordarray[2]) == {"one": 3, "two": [4.4, 5.5]}

def test_type():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray()
    recordarray.append(content1)
    recordarray.append(listoffsetarray)
    assert str(awkward1.typeof(recordarray)) == '5 * (int64, var * float64)'

    assert awkward1.typeof(recordarray) == awkward1.layout.ArrayType(awkward1.layout.RecordType(
        awkward1.layout.PrimitiveType("int64"),
        awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64"))), 5)
    assert awkward1.typeof(recordarray).compatible(awkward1.layout.ArrayType(awkward1.layout.RecordType(
        awkward1.layout.PrimitiveType("int64"),
        awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64"))), 5))
    assert awkward1.typeof(recordarray[2]) == awkward1.layout.RecordType(
        awkward1.layout.PrimitiveType("int64"),
        awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64")))
    assert awkward1.typeof(recordarray[2]).compatible(awkward1.layout.RecordType(
        awkward1.layout.PrimitiveType("int64"),
        awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64"))))

    recordarray.setkey(0, "one")
    recordarray.setkey(1, "two")
    assert str(awkward1.typeof(recordarray)) in ('5 * {"one": int64, "two": var * float64}', '5 * {"two": var * float64, "one": int64}')

    assert str(awkward1.layout.RecordType(
        awkward1.layout.PrimitiveType("int32"),
        awkward1.layout.PrimitiveType("float64"))) == '(int32, float64)'

    assert str(awkward1.layout.RecordType(
        one=awkward1.layout.PrimitiveType("int32"),
        two=awkward1.layout.PrimitiveType("float64"))) in ('{"one": int32, "two": float64}', '{"two": float64, "one": int32}')

    assert awkward1.typeof(recordarray) == awkward1.layout.ArrayType(awkward1.layout.RecordType(
        one=awkward1.layout.PrimitiveType("int64"),
        two=awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64"))), 5)
    assert awkward1.typeof(recordarray).compatible(awkward1.layout.ArrayType(awkward1.layout.RecordType(
        one=awkward1.layout.PrimitiveType("int64"),
        two=awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64"))), 5))
    assert awkward1.typeof(recordarray[2]) == awkward1.layout.RecordType(
        one=awkward1.layout.PrimitiveType("int64"),
        two=awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64")))
    assert awkward1.typeof(recordarray[2]).compatible(awkward1.layout.RecordType(
        one=awkward1.layout.PrimitiveType("int64"),
        two=awkward1.layout.ListType(awkward1.layout.PrimitiveType("float64"))))

def test_getitem():
    assert str(awkward1.layout.Slice((1, 2, [3], "four", ["five", "six"], slice(7, 8, 9)))) == '[array([1]), array([2]), array([3]), "four", ["five", "six"], 7:8:9]'

    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray, content2])
    assert recordarray.istuple

    assert awkward1.tolist(recordarray["2"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(recordarray[["0", "1"]]) == [{"0": 1, "1": [1.1, 2.2, 3.3]}, {"0": 2, "1": []}, {"0": 3, "1": [4.4, 5.5]}, {"0": 4, "1": [6.6]}, {"0": 5, "1": [7.7, 8.8, 9.9]}]
    assert awkward1.tolist(recordarray[["1", "0"]]) == [{"1": 1, "0": [1.1, 2.2, 3.3]}, {"1": 2, "0": []}, {"1": 3, "0": [4.4, 5.5]}, {"1": 4, "0": [6.6]}, {"1": 5, "0": [7.7, 8.8, 9.9]}]
    assert awkward1.tolist(recordarray[1:-1]) == [{"0": 2, "1": [], "2": 2.2}, {"0": 3, "1": [4.4, 5.5], "2": 3.3}, {"0": 4, "1": [6.6], "2": 4.4}]
    assert awkward1.tolist(recordarray[2]) == {"0": 3, "1": [4.4, 5.5], "2": 3.3}
    assert awkward1.tolist(recordarray[2]["1"]) == [4.4, 5.5]
    assert awkward1.tolist(recordarray[2][["0", "1"]]) == {"0": 3, "1": [4.4, 5.5]}
    assert awkward1.tolist(recordarray[2][["1", "0"]]) == {"1": 3, "0": [4.4, 5.5]}

    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray, "three": content2})
    assert not recordarray.istuple

    assert awkward1.tolist(recordarray["three"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(recordarray[["one", "two"]]) == [{"one": 1, "two": [1.1, 2.2, 3.3]}, {"one": 2, "two": []}, {"one": 3, "two": [4.4, 5.5]}, {"one": 4, "two": [6.6]}, {"one": 5, "two": [7.7, 8.8, 9.9]}]
    assert awkward1.tolist(recordarray[["two", "one"]]) == [{"one": 1, "two": [1.1, 2.2, 3.3]}, {"one": 2, "two": []}, {"one": 3, "two": [4.4, 5.5]}, {"one": 4, "two": [6.6]}, {"one": 5, "two": [7.7, 8.8, 9.9]}]
    assert awkward1.tolist(recordarray[1:-1]) == [{"one": 2, "two": [], "three": 2.2}, {"one": 3, "two": [4.4, 5.5], "three": 3.3}, {"one": 4, "two": [6.6], "three": 4.4}]
    assert awkward1.tolist(recordarray[2]) == {"one": 3, "two": [4.4, 5.5], "three": 3.3}
    assert awkward1.tolist(recordarray[2]["two"]) == [4.4, 5.5]
    assert awkward1.tolist(recordarray[2][["one", "two"]]) == {"one": 3, "two": [4.4, 5.5]}
    assert awkward1.tolist(recordarray[2][["two", "one"]]) == {"one": 3, "two": [4.4, 5.5]}

def test_getitem_other_types():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content2)
    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray1, "three": content2})

    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, recordarray)
    assert awkward1.tolist(listoffsetarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert awkward1.tolist(listoffsetarray2["two"]) == [[[1.1, 2.2, 3.3], [], [4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert awkward1.tolist(listoffsetarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.tolist(listoffsetarray2[["two", "three"]]) == [[{"two": [1.1, 2.2, 3.3], "three": 1.1}, {"two": [], "three": 2.2}, {"two": [4.4, 5.5], "three": 3.3}], [], [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}]]

    starts2 = awkward1.layout.Index64(numpy.array([0, 3, 3]))
    stops2 = awkward1.layout.Index64(numpy.array([3, 3, 5]))
    listarray2 = awkward1.layout.ListArray64(starts2, stops2, recordarray)
    assert awkward1.tolist(listarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert awkward1.tolist(listarray2["two"]) == [[[1.1, 2.2, 3.3], [], [4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert awkward1.tolist(listarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.tolist(listarray2[["two", "three"]]) == [[{"two": [1.1, 2.2, 3.3], "three": 1.1}, {"two": [], "three": 2.2}, {"two": [4.4, 5.5], "three": 3.3}], [], [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}]]

    regulararray2 = awkward1.layout.RegularArray(recordarray, 1)
    assert awkward1.tolist(regulararray2["one"]) == [[1], [2], [3], [4], [5]]
    assert awkward1.tolist(regulararray2["two"]) == [[[1.1, 2.2, 3.3]], [[]], [[4.4, 5.5]], [[6.6]], [[7.7, 8.8, 9.9]]]
    assert awkward1.tolist(regulararray2["three"]) == [[1.1], [2.2], [3.3], [4.4], [5.5]]
    assert awkward1.tolist(regulararray2[["two", "three"]]) == [[{"two": [1.1, 2.2, 3.3], "three": 1.1}], [{"two": [], "three": 2.2}], [{"two": [4.4, 5.5], "three": 3.3}], [{"two": [6.6], "three": 4.4}], [{"two": [7.7, 8.8, 9.9], "three": 5.5}]]

def test_getitem_next():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content2)
    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray1, "three": content2})
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, recordarray)

    assert awkward1.tolist(listoffsetarray2) == [[{"one": 1, "two": [1.1, 2.2, 3.3], "three": 1.1}, {"one": 2, "two": [], "three": 2.2}, {"one": 3, "two": [4.4, 5.5], "three": 3.3}], [], [{"one": 4, "two": [6.6], "three": 4.4}, {"one": 5, "two": [7.7, 8.8, 9.9], "three": 5.5}]]

    assert awkward1.tolist(listoffsetarray2[2, "one"]) == [4, 5]
    assert awkward1.tolist(listoffsetarray2[2, "two"]) == [[6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(listoffsetarray2[2, "three"]) == [4.4, 5.5]
    assert awkward1.tolist(listoffsetarray2[2, ["two", "three"]]) == [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}]
