# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

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
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray, content2, content1], keys=["one", "two", "2", "wonky"])
    assert awkward1.to_list(recordarray.field(0)) == [1, 2, 3, 4, 5]
    assert awkward1.to_list(recordarray.field("two")) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(recordarray.field("wonky")) == [1, 2, 3, 4, 5]

    str(recordarray)
    assert awkward1.to_json(recordarray) == '[{"one":1,"two":[1.1,2.2,3.3],"2":1.1,"wonky":1},{"one":2,"two":[],"2":2.2,"wonky":2},{"one":3,"two":[4.4,5.5],"2":3.3,"wonky":3},{"one":4,"two":[6.6],"2":4.4,"wonky":4},{"one":5,"two":[7.7,8.8,9.9],"2":5.5,"wonky":5}]'

    assert len(recordarray) == 5
    assert recordarray.key(0) == "one"
    assert recordarray.key(1) == "two"
    assert recordarray.key(2) == "2"
    assert recordarray.key(3) == "wonky"
    assert recordarray.fieldindex("wonky") == 3
    assert recordarray.fieldindex("one") == 0
    assert recordarray.fieldindex("0") == 0
    assert recordarray.fieldindex("two") == 1
    assert recordarray.fieldindex("1") == 1
    assert recordarray.fieldindex("2") == 2
    assert recordarray.haskey("wonky")
    assert recordarray.haskey("one")
    assert recordarray.haskey("0")
    assert recordarray.haskey("two")
    assert recordarray.haskey("1")
    assert recordarray.haskey("2")

    assert recordarray.keys() == ["one", "two", "2", "wonky"]
    assert [awkward1.to_list(x) for x in recordarray.fields()] == [[1, 2, 3, 4, 5], [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [1, 2, 3, 4, 5]]
    pairs = recordarray.fielditems()
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[2][0] == "2"
    assert pairs[3][0] == "wonky"
    assert awkward1.to_list(pairs[0][1]) == [1, 2, 3, 4, 5]
    assert awkward1.to_list(pairs[1][1]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(pairs[2][1]) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert awkward1.to_list(pairs[3][1]) == [1, 2, 3, 4, 5]

    assert awkward1.to_json(recordarray.astuple) == '[{"0":1,"1":[1.1,2.2,3.3],"2":1.1,"3":1},{"0":2,"1":[],"2":2.2,"3":2},{"0":3,"1":[4.4,5.5],"2":3.3,"3":3},{"0":4,"1":[6.6],"2":4.4,"3":4},{"0":5,"1":[7.7,8.8,9.9],"2":5.5,"3":5}]'

def test_scalar_record():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray], keys=["one", "two"])

    str(recordarray)
    str(recordarray[2])
    assert awkward1.to_json(recordarray[2]) == '{"one":3,"two":[4.4,5.5]}'

    assert recordarray[2].keys() == ["one", "two"]
    assert [awkward1.to_list(x) for x in recordarray[2].fields()] == [3, [4.4, 5.5]]
    pairs = recordarray[2].fielditems()
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[0][1] == 3
    assert awkward1.to_list(pairs[1][1]) == [4.4, 5.5]
    assert awkward1.to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5]}

    assert awkward1.to_list(awkward1.layout.Record(recordarray, 2)) == {"one": 3, "two": [4.4, 5.5]}

def test_type():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray])
    assert str(awkward1.type(recordarray)) == '(int64, var * float64)'

    assert awkward1.type(recordarray) == awkward1.types.RecordType((
        awkward1.types.PrimitiveType("int64"),
        awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))))
    assert awkward1.type(recordarray[2]) == awkward1.types.RecordType(
        (awkward1.types.PrimitiveType("int64"),
        awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))))

    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray], keys=["one", "two"])
    assert str(awkward1.type(recordarray)) in ('{"one": int64, "two": var * float64}', '{"two": var * float64, "one": int64}')

    assert str(awkward1.types.RecordType(
        (awkward1.types.PrimitiveType("int32"),
        awkward1.types.PrimitiveType("float64")))) == '(int32, float64)'

    assert str(awkward1.types.RecordType(
        {"one": awkward1.types.PrimitiveType("int32"),
        "two": awkward1.types.PrimitiveType("float64")})) in ('{"one": int32, "two": float64}', '{"two": float64, "one": int32}')

    assert awkward1.type(recordarray) == awkward1.types.RecordType({
        "one": awkward1.types.PrimitiveType("int64"),
        "two": awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))})
    assert awkward1.type(recordarray[2]) == awkward1.types.RecordType({
        "one": awkward1.types.PrimitiveType("int64"),
        "two": awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))})

def test_getitem():
    assert awkward1._ext._slice_tostring((1, 2, [3], "four", ["five", "six"], slice(7, 8, 9))) == '[array([1]), array([2]), array([3]), "four", ["five", "six"], 7:8:9]'

    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray, content2])
    assert recordarray.istuple

    assert awkward1.to_list(recordarray["2"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(recordarray[["0", "1"]]) == [(1, [1.1, 2.2, 3.3]), (2, []), (3, [4.4, 5.5]), (4, [6.6]), (5, [7.7, 8.8, 9.9])]
    assert awkward1.to_list(recordarray[["1", "0"]]) == [([1.1, 2.2, 3.3], 1), ([], 2), ([4.4, 5.5], 3), ([6.6], 4), ([7.7, 8.8, 9.9], 5)]
    assert awkward1.to_list(recordarray[1:-1]) == [(2, [], 2.2), (3, [4.4, 5.5], 3.3), (4, [6.6], 4.4)]
    assert awkward1.to_list(recordarray[2]) == (3, [4.4, 5.5], 3.3)
    assert awkward1.to_list(recordarray[2]["1"]) == [4.4, 5.5]
    assert awkward1.to_list(recordarray[2][["0", "1"]]) == (3, [4.4, 5.5])
    assert awkward1.to_list(recordarray[2][["1", "0"]]) == ([4.4, 5.5], 3)

    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray, "three": content2})
    assert not recordarray.istuple

    assert awkward1.to_list(recordarray["three"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.to_list(recordarray[["one", "two"]]) == [{"one": 1, "two": [1.1, 2.2, 3.3]}, {"one": 2, "two": []}, {"one": 3, "two": [4.4, 5.5]}, {"one": 4, "two": [6.6]}, {"one": 5, "two": [7.7, 8.8, 9.9]}]
    assert awkward1.to_list(recordarray[["two", "one"]]) == [{"one": 1, "two": [1.1, 2.2, 3.3]}, {"one": 2, "two": []}, {"one": 3, "two": [4.4, 5.5]}, {"one": 4, "two": [6.6]}, {"one": 5, "two": [7.7, 8.8, 9.9]}]
    assert awkward1.to_list(recordarray[1:-1]) == [{"one": 2, "two": [], "three": 2.2}, {"one": 3, "two": [4.4, 5.5], "three": 3.3}, {"one": 4, "two": [6.6], "three": 4.4}]
    assert awkward1.to_list(recordarray[2]) == {"one": 3, "two": [4.4, 5.5], "three": 3.3}
    assert awkward1.to_list(recordarray[2]["two"]) == [4.4, 5.5]
    assert awkward1.to_list(recordarray[2][["one", "two"]]) == {"one": 3, "two": [4.4, 5.5]}
    assert awkward1.to_list(recordarray[2][["two", "one"]]) == {"one": 3, "two": [4.4, 5.5]}

def test_getitem_other_types():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content2)
    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray1, "three": content2})

    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, recordarray)
    assert awkward1.to_list(listoffsetarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert awkward1.to_list(listoffsetarray2["two"]) == [[[1.1, 2.2, 3.3], [], [4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert awkward1.to_list(listoffsetarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(listoffsetarray2[["two", "three"]]) == [[{"two": [1.1, 2.2, 3.3], "three": 1.1}, {"two": [], "three": 2.2}, {"two": [4.4, 5.5], "three": 3.3}], [], [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}]]

    starts2 = awkward1.layout.Index64(numpy.array([0, 3, 3]))
    stops2 = awkward1.layout.Index64(numpy.array([3, 3, 5]))
    listarray2 = awkward1.layout.ListArray64(starts2, stops2, recordarray)
    assert awkward1.to_list(listarray2["one"]) == [[1, 2, 3], [], [4, 5]]
    assert awkward1.to_list(listarray2["two"]) == [[[1.1, 2.2, 3.3], [], [4.4, 5.5]], [], [[6.6], [7.7, 8.8, 9.9]]]
    assert awkward1.to_list(listarray2["three"]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    assert awkward1.to_list(listarray2[["two", "three"]]) == [[{"two": [1.1, 2.2, 3.3], "three": 1.1}, {"two": [], "three": 2.2}, {"two": [4.4, 5.5], "three": 3.3}], [], [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}]]

    regulararray2 = awkward1.layout.RegularArray(recordarray, 1)
    assert awkward1.to_list(regulararray2["one"]) == [[1], [2], [3], [4], [5]]
    assert awkward1.to_list(regulararray2["two"]) == [[[1.1, 2.2, 3.3]], [[]], [[4.4, 5.5]], [[6.6]], [[7.7, 8.8, 9.9]]]
    assert awkward1.to_list(regulararray2["three"]) == [[1.1], [2.2], [3.3], [4.4], [5.5]]
    assert awkward1.to_list(regulararray2[["two", "three"]]) == [[{"two": [1.1, 2.2, 3.3], "three": 1.1}], [{"two": [], "three": 2.2}], [{"two": [4.4, 5.5], "three": 3.3}], [{"two": [6.6], "three": 4.4}], [{"two": [7.7, 8.8, 9.9], "three": 5.5}]]

def test_getitem_next():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    content3 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=numpy.float64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content2)
    listoffsetarray3 = awkward1.layout.ListOffsetArray64(offsets1, content3)
    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray1, "three": content2, "four": listoffsetarray3})
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, recordarray)

    assert awkward1.to_list(listoffsetarray2[2, "one"]) == [4, 5]
    assert awkward1.to_list(listoffsetarray2[2, "two"]) == [[6.6], [7.7, 8.8, 9.9]]
    assert awkward1.to_list(listoffsetarray2[2, "three"]) == [4.4, 5.5]
    assert awkward1.to_list(listoffsetarray2[2, ["two", "three"]]) == [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}]

    assert awkward1.to_list(listoffsetarray2[2, 1]) == {"one": 5, "two": [7.7, 8.8, 9.9], "three": 5.5, "four": [7, 8, 9]}
    with pytest.raises(ValueError):
        listoffsetarray2[2, 1, 0]
    assert listoffsetarray2[2, 1, "one"] == 5
    assert awkward1.to_list(listoffsetarray2[2, 1, "two"]) == [7.7, 8.8, 9.9]
    assert listoffsetarray2[2, 1, "two", 1] == 8.8
    assert awkward1.to_list(listoffsetarray2[2, 1, ["two", "four"], 1]) == {"two": 8.8, "four": 8}
    assert awkward1.to_list(listoffsetarray2[2, 1, ["two", "four"], 1:]) == {"two": [8.8, 9.9], "four": [8, 9]}

def test_setidentities():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)

    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray])
    recordarray.setidentities()

    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray})
    recordarray.setidentities()
    assert recordarray["one"].identities.fieldloc == [(0, "one")]
    assert recordarray["two"].identities.fieldloc == [(0, "two")]
    assert recordarray["one", 1] == 2
    assert recordarray[1, "one"] == 2
    assert recordarray["two", 2, 1] == 5.5
    assert recordarray[2, "two", 1] == 5.5

    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray})
    recordarray2 = awkward1.layout.RecordArray({"outer": recordarray})
    recordarray2.setidentities()
    assert recordarray2["outer"].identities.fieldloc == [(0, "outer")]
    assert recordarray2["outer", "one"].identities.fieldloc == [(0, "outer"), (0, "one")]
    assert recordarray2["outer", "two"].identities.fieldloc == [(0, "outer"), (0, "two")]
    assert recordarray2["outer", "one", 1] == 2
    assert recordarray2["outer", 1, "one"] == 2
    assert recordarray2[1, "outer", "one"] == 2
    assert recordarray2["outer", "two", 2, 1] == 5.5
    assert recordarray2["outer", 2, "two", 1] == 5.5
    assert recordarray2[2, "outer", "two", 1] == 5.5
    with pytest.raises(ValueError) as excinfo:
        recordarray2["outer", "two", 0, 99]
    assert ' with identity [0, "outer", "two"] attempting to get 99, index out of range' in str(excinfo.value)
    assert recordarray2.identity == ()
    assert recordarray2[2].identity == (2,)
    assert recordarray2[2, "outer"].identity == (2, "outer")
    assert recordarray2[2, "outer", "two"].identity == (2, "outer", "two")

    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray})
    recordarray2 = awkward1.layout.RecordArray({"outer": awkward1.layout.RegularArray(recordarray, 1)})
    recordarray2.setidentities()
    assert recordarray2["outer"].identities.fieldloc == [(0, "outer")]

    assert recordarray2["outer", 0, "one"].content.identities.fieldloc == [(0, "outer"), (1, "one")]
    assert recordarray2["outer", 0, "two"].content.identities.fieldloc == [(0, "outer"), (1, "two")]
    assert recordarray2["outer", "one", 0].identities.fieldloc == [(0, "outer"), (1, "one")]
    assert recordarray2["outer", "two", 0].identities.fieldloc == [(0, "outer"), (1, "two")]
    assert recordarray2["outer", "one", 1, 0] == 2
    assert recordarray2["outer", 1, "one", 0] == 2
    assert recordarray2["outer", 1, 0, "one"] == 2
    assert recordarray2[1, "outer", "one", 0] == 2
    assert recordarray2[1, "outer", 0, "one"] == 2
    assert recordarray2[1, 0, "outer", "one"] == 2

    with pytest.raises(ValueError) as excinfo:
        recordarray2["outer", 2, "two", 0, 99]
    assert ' with identity [2, "outer", 0, "two"] attempting to get 99, index out of range' in str(excinfo.value)
    assert recordarray2.identity == ()
    assert recordarray2[2].identity == (2,)
    assert recordarray2[2, "outer"].identity == (2, "outer")
    assert recordarray2[2, "outer", 0].identity == (2, "outer", 0)
    assert recordarray2[2, "outer", 0, "two"].identity == (2, "outer", 0, "two")

def test_identities():
    a = awkward1.from_iter([
        {"outer": [{"x": 0, "y": []}, {"x": 1, "y": [1]}, {"x": 2, "y": [1, 2]}]},
        {"outer": []},
        {"outer": [{"x": 3, "y": [1, 2, 3]}, {"x": 4, "y": [1, 2, 3, 4]}]}
        ], highlevel=False)
    a.setidentities()

    b = awkward1.from_iter([
        {"outer": {"x": 0, "y": []}},
        {"outer": {"x": 1, "y": [1]}},
        {"outer": {"x": 2, "y": [1, 2]}},
        {"outer": {"x": 3, "y": [1, 2, 3]}},
        {"outer": {"x": 4, "y": [1, 2, 3, 4]}}
    ], highlevel=False)
    b.setidentities()

    assert awkward1.to_list(a) == [{u'outer': [{u'x': 0, u'y': []},
                                               {u'x': 1, u'y': [1]},
                                               {u'x': 2, u'y': [1, 2]}]},
                                   {u'outer': []},
                                   {u'outer': [{u'x': 3, u'y': [1, 2, 3]},
                                               {u'x': 4, u'y': [1, 2, 3, 4]}]}]
    assert a.identity == ()
    assert a[2].identity == (2,)
    assert a[2, "outer"].identity == (2, 'outer')
    assert a[2, "outer", 0].identity == (2, 'outer', 0)
    assert a[2, "outer", 0, "y"].identity == (2, u'outer', 0, u'y')

    assert awkward1.to_list(b) == [{u'outer': {u'x': 0, u'y': []}},
                                   {u'outer': {u'x': 1, u'y': [1]}},
                                   {u'outer': {u'x': 2, u'y': [1, 2]}},
                                   {u'outer': {u'x': 3, u'y': [1, 2, 3]}},
                                   {u'outer': {u'x': 4, u'y': [1, 2, 3, 4]}}]
    assert b.identity == ()
    assert b[2].identity == (2,)
    assert b[2, "outer"].identity == (2, u'outer')
    with pytest.raises(ValueError) as err:
        b[2, "outer", 0].identity
    assert str(err.value).startswith("in NumpyArray, too many dimensions in slice")
    with pytest.raises(ValueError) as err:
        b[2, "outer", 0, "y"].identity
    assert str(err.value).startswith("in NumpyArray, too many dimensions in slice")

def test_builder_tuple():
    typestrs = {}
    builder = awkward1.layout.ArrayBuilder()
    assert str(builder.type(typestrs)) == 'unknown'
    assert awkward1.to_list(builder.snapshot()) == []

    builder.begintuple(0)
    builder.endtuple()

    builder.begintuple(0)
    builder.endtuple()

    builder.begintuple(0)
    builder.endtuple()

    assert str(builder.type(typestrs)) == '()'
    assert awkward1.to_list(builder.snapshot()) == [(), (), ()]

    builder = awkward1.layout.ArrayBuilder()

    builder.begintuple(3)
    builder.index(0)
    builder.boolean(True)
    builder.index(1)
    builder.beginlist()
    builder.integer(1)
    builder.endlist()
    builder.index(2)
    builder.real(1.1)
    builder.endtuple()

    builder.begintuple(3)
    builder.index(1)
    builder.beginlist()
    builder.integer(2)
    builder.integer(2)
    builder.endlist()
    builder.index(2)
    builder.real(2.2)
    builder.index(0)
    builder.boolean(False)
    builder.endtuple()

    builder.begintuple(3)
    builder.index(2)
    builder.real(3.3)
    builder.index(1)
    builder.beginlist()
    builder.integer(3)
    builder.integer(3)
    builder.integer(3)
    builder.endlist()
    builder.index(0)
    builder.boolean(True)
    builder.endtuple()

    assert str(builder.type(typestrs)) == '(bool, var * int64, float64)'
    assert awkward1.to_list(builder.snapshot()) == [(True, [1], 1.1), (False, [2, 2], 2.2), (True, [3, 3, 3], 3.3)]

def test_builder_record():
    typestrs = {}
    builder = awkward1.layout.ArrayBuilder()
    assert str(builder.type(typestrs)) == 'unknown'
    assert awkward1.to_list(builder.snapshot()) == []

    builder.beginrecord()
    builder.endrecord()

    builder.beginrecord()
    builder.endrecord()

    builder.beginrecord()
    builder.endrecord()

    assert str(builder.type(typestrs)) == '{}'
    assert awkward1.to_list(builder.snapshot()) == [{}, {}, {}]

    builder = awkward1.layout.ArrayBuilder()

    builder.beginrecord()
    builder.field("one")
    builder.integer(1)
    builder.field("two")
    builder.real(1.1)
    builder.endrecord()

    builder.beginrecord()
    builder.field("two")
    builder.real(2.2)
    builder.field("one")
    builder.integer(2)
    builder.endrecord()

    builder.beginrecord()
    builder.field("one")
    builder.integer(3)
    builder.field("two")
    builder.real(3.3)
    builder.endrecord()

    assert str(builder.type(typestrs)) == '{"one": int64, "two": float64}'
    assert awkward1.to_list(builder.snapshot()) == [{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}]

def test_fromiter():
    dataset = [
        [(1, 1.1), (2, 2.2), (3, 3.3)],
        [(1, [1.1, 2.2, 3.3]), (2, []), (3, [4.4, 5.5])],
        [[(1, 1.1), (2, 2.2), (3, 3.3)], [], [(4, 4.4), (5, 5.5)]],
        [((1, 1), 1.1), ((2, 2), 2.2), ((3, 3), 3.3)],
        [({"x": 1, "y": 1}, 1.1), ({"x": 2, "y": 2}, 2.2), ({"x": 3, "y": 3}, 3.3)],
        [{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}],
        [{"one": 1, "two": [1.1, 2.2, 3.3]}, {"one": 2, "two": []}, {"one": 3, "two": [4.4, 5.5]}],
        [[{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}], [], [{"one": 4, "two": 4.4}, {"one": 5, "two": 5.5}]],
        [{"one": {"x": 1, "y": 1}, "two": 1.1}, {"one": {"x": 2, "y": 2}, "two": 2.2}, {"one": {"x": 3, "y": 3}, "two": 3.3}],
        [{"one": (1, 1), "two": 1.1}, {"one": (2, 2), "two": 2.2}, {"one": (3, 3), "two": 3.3}],
    ]
    for datum in dataset:
        assert awkward1.to_list(awkward1.from_iter(datum)) == datum

def test_json():
    dataset = [
        '[{"one":1,"two":1.1},{"one":2,"two":2.2},{"one":3,"two":3.3}]',
        '[{"one":1,"two":[1.1,2.2,3.3]},{"one":2,"two":[]},{"one":3,"two":[4.4,5.5]}]',
        '[[{"one":1,"two":1.1},{"one":2,"two":2.2},{"one":3,"two":3.3}],[],[{"one":4,"two":4.4},{"one":5,"two":5.5}]]',
        '[{"one":{"x":1,"y":1},"two":1.1},{"one":{"x":2,"y":2},"two":2.2},{"one":{"x":3,"y":3},"two":3.3}]',
    ]
    for datum in dataset:
        assert awkward1.to_json(awkward1.from_json(datum)) == datum
