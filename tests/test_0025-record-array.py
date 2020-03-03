# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

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
    assert awkward1.tolist(recordarray.field(0)) == [1, 2, 3, 4, 5]
    assert awkward1.tolist(recordarray.field("two")) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(recordarray.field("wonky")) == [1, 2, 3, 4, 5]

    str(recordarray)
    assert awkward1.tojson(recordarray) == '[{"one":1,"two":[1.1,2.2,3.3],"2":1.1,"wonky":1},{"one":2,"two":[],"2":2.2,"wonky":2},{"one":3,"two":[4.4,5.5],"2":3.3,"wonky":3},{"one":4,"two":[6.6],"2":4.4,"wonky":4},{"one":5,"two":[7.7,8.8,9.9],"2":5.5,"wonky":5}]'

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
    assert [awkward1.tolist(x) for x in recordarray.fields()] == [[1, 2, 3, 4, 5], [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], [1, 2, 3, 4, 5]]
    pairs = recordarray.fielditems()
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[2][0] == "2"
    assert pairs[3][0] == "wonky"
    assert awkward1.tolist(pairs[0][1]) == [1, 2, 3, 4, 5]
    assert awkward1.tolist(pairs[1][1]) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(pairs[2][1]) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert awkward1.tolist(pairs[3][1]) == [1, 2, 3, 4, 5]

    assert awkward1.tojson(recordarray.astuple) == '[{"0":1,"1":[1.1,2.2,3.3],"2":1.1,"3":1},{"0":2,"1":[],"2":2.2,"3":2},{"0":3,"1":[4.4,5.5],"2":3.3,"3":3},{"0":4,"1":[6.6],"2":4.4,"3":4},{"0":5,"1":[7.7,8.8,9.9],"2":5.5,"3":5}]'

def test_scalar_record():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray], keys=["one", "two"])

    str(recordarray)
    str(recordarray[2])
    assert awkward1.tojson(recordarray[2]) == '{"one":3,"two":[4.4,5.5]}'

    assert recordarray[2].keys() == ["one", "two"]
    assert [awkward1.tolist(x) for x in recordarray[2].fields()] == [3, [4.4, 5.5]]
    pairs = recordarray[2].fielditems()
    assert pairs[0][0] == "one"
    assert pairs[1][0] == "two"
    assert pairs[0][1] == 3
    assert awkward1.tolist(pairs[1][1]) == [4.4, 5.5]
    assert awkward1.tolist(recordarray[2]) == {"one": 3, "two": [4.4, 5.5]}

    assert awkward1.tolist(awkward1.layout.Record(recordarray, 2)) == {"one": 3, "two": [4.4, 5.5]}

def test_type():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray])
    assert str(awkward1.typeof(recordarray)) == '(int64, var * float64)'

    assert awkward1.typeof(recordarray) == awkward1.types.RecordType((
        awkward1.types.PrimitiveType("int64"),
        awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))))
    assert awkward1.typeof(recordarray[2]) == awkward1.types.RecordType(
        (awkward1.types.PrimitiveType("int64"),
        awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))))

    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray], keys=["one", "two"])
    assert str(awkward1.typeof(recordarray)) in ('{"one": int64, "two": var * float64}', '{"two": var * float64, "one": int64}')

    assert str(awkward1.types.RecordType(
        (awkward1.types.PrimitiveType("int32"),
        awkward1.types.PrimitiveType("float64")))) == '(int32, float64)'

    assert str(awkward1.types.RecordType(
        {"one": awkward1.types.PrimitiveType("int32"),
        "two": awkward1.types.PrimitiveType("float64")})) in ('{"one": int32, "two": float64}', '{"two": float64, "one": int32}')

    assert awkward1.typeof(recordarray) == awkward1.types.RecordType({
        "one": awkward1.types.PrimitiveType("int64"),
        "two": awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))})
    assert awkward1.typeof(recordarray[2]) == awkward1.types.RecordType({
        "one": awkward1.types.PrimitiveType("int64"),
        "two": awkward1.types.ListType(awkward1.types.PrimitiveType("float64"))})

def test_getitem():
    assert awkward1.layout._slice_tostring((1, 2, [3], "four", ["five", "six"], slice(7, 8, 9))) == '[array([1]), array([2]), array([3]), "four", ["five", "six"], 7:8:9]'

    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], dtype=numpy.float64))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray([content1, listoffsetarray, content2])
    assert recordarray.istuple

    assert awkward1.tolist(recordarray["2"]) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(recordarray[["0", "1"]]) == [(1, [1.1, 2.2, 3.3]), (2, []), (3, [4.4, 5.5]), (4, [6.6]), (5, [7.7, 8.8, 9.9])]
    assert awkward1.tolist(recordarray[["1", "0"]]) == [([1.1, 2.2, 3.3], 1), ([], 2), ([4.4, 5.5], 3), ([6.6], 4), ([7.7, 8.8, 9.9], 5)]
    assert awkward1.tolist(recordarray[1:-1]) == [(2, [], 2.2), (3, [4.4, 5.5], 3.3), (4, [6.6], 4.4)]
    assert awkward1.tolist(recordarray[2]) == (3, [4.4, 5.5], 3.3)
    assert awkward1.tolist(recordarray[2]["1"]) == [4.4, 5.5]
    assert awkward1.tolist(recordarray[2][["0", "1"]]) == (3, [4.4, 5.5])
    assert awkward1.tolist(recordarray[2][["1", "0"]]) == ([4.4, 5.5], 3)

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
    content3 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=numpy.float64))
    offsets1 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray1 = awkward1.layout.ListOffsetArray64(offsets1, content2)
    listoffsetarray3 = awkward1.layout.ListOffsetArray64(offsets1, content3)
    recordarray = awkward1.layout.RecordArray({"one": content1, "two": listoffsetarray1, "three": content2, "four": listoffsetarray3})
    offsets2 = awkward1.layout.Index64(numpy.array([0, 3, 3, 5]))
    listoffsetarray2 = awkward1.layout.ListOffsetArray64(offsets2, recordarray)

    assert awkward1.tolist(listoffsetarray2[2, "one"]) == [4, 5]
    assert awkward1.tolist(listoffsetarray2[2, "two"]) == [[6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(listoffsetarray2[2, "three"]) == [4.4, 5.5]
    assert awkward1.tolist(listoffsetarray2[2, ["two", "three"]]) == [{"two": [6.6], "three": 4.4}, {"two": [7.7, 8.8, 9.9], "three": 5.5}]

    assert awkward1.tolist(listoffsetarray2[2, 1]) == {"one": 5, "two": [7.7, 8.8, 9.9], "three": 5.5, "four": [7, 8, 9]}
    with pytest.raises(ValueError):
        listoffsetarray2[2, 1, 0]
    assert listoffsetarray2[2, 1, "one"] == 5
    assert awkward1.tolist(listoffsetarray2[2, 1, "two"]) == [7.7, 8.8, 9.9]
    assert listoffsetarray2[2, 1, "two", 1] == 8.8
    assert awkward1.tolist(listoffsetarray2[2, 1, ["two", "four"], 1]) == {"two": 8.8, "four": 8}
    assert awkward1.tolist(listoffsetarray2[2, 1, ["two", "four"], 1:]) == {"two": [8.8, 9.9], "four": [8, 9]}

content1_a = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
content2_a = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
offsets_a = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
listoffsetarray_a = awkward1.layout.ListOffsetArray64(offsets_a, content2_a)
recordarray_a = awkward1.layout.RecordArray([content1_a, listoffsetarray_a])
recordarray_a.setidentities()

# content1_b = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
# content2_b = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
# offsets_b = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
# listoffsetarray_b = awkward1.layout.ListOffsetArray64(offsets_b, content2_b)
# recordarray_b = awkward1.layout.RecordArray({"one": content1_b, "two": listoffsetarray_b})
# recordarray_b.setidentities()
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_1():
    assert recordarray_b["one"].identities.fieldloc == [(0, "one")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_2():
    assert recordarray_b["two"].identities.fieldloc == [(0, "two")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_3():
    assert recordarray_b["one", 1] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_4():
    assert recordarray_b[1, "one"] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_5():
    assert recordarray_b["two", 2, 1] == 5.5
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_6():
    assert recordarray_b[2, "two", 1] == 5.5

# content1_c = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
# content2_c = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
# offsets_c = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
# listoffsetarray_c = awkward1.layout.ListOffsetArray64(offsets_c, content2_c)
# recordarray_c = awkward1.layout.RecordArray({"one": content1_c, "two": listoffsetarray_c})
# recordarray2_c = awkward1.layout.RecordArray({"outer": recordarray_c})
# recordarray2_c.setidentities()
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_7():
    assert recordarray2_c["outer"].identities.fieldloc == [(0, "outer")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_8():
    assert recordarray2_c["outer", "one"].identities.fieldloc == [(0, "outer"), (0, "one")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_9():
    assert recordarray2_c["outer", "two"].identities.fieldloc == [(0, "outer"), (0, "two")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_10():
    assert recordarray2_c["outer", "one", 1] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_11():
    assert recordarray2_c["outer", 1, "one"] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_12():
    assert recordarray2_c[1, "outer", "one"] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_13():
    assert recordarray2_c["outer", "two", 2, 1] == 5.5
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_14():
    assert recordarray2_c["outer", 2, "two", 1] == 5.5
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_15():
    assert recordarray2_c[2, "outer", "two", 1] == 5.5
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_16():
    with pytest.raises(ValueError) as excinfo:
        recordarray2_c["outer", "two", 0, 99]
    assert str(excinfo.value) == 'in ListArray64 with identity [0, "outer", "two"] attempting to get 99, index out of range'
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_17():
    assert recordarray2_c.identity == ()
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_18():
    assert recordarray2_c[2].identity == (2,)
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_19():
    assert recordarray2_c[2, "outer"].identity == (2, "outer")
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_20():
    assert recordarray2_c[2, "outer", "two"].identity == (2, "outer", "two")

# content1_d = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
# content2_d = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
# offsets_d = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
# listoffsetarray_d = awkward1.layout.ListOffsetArray64(offsets_d, content2_d)
# recordarray_d = awkward1.layout.RecordArray({"one": content1_d, "two": listoffsetarray_d})
# recordarray2_d = awkward1.layout.RecordArray({"outer": awkward1.layout.RegularArray(recordarray_d, 1)})
# recordarray2_d.setidentities()
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_21():
    assert recordarray2_d["outer"].identities.fieldloc == [(0, "outer")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_22():
    assert recordarray2_d["outer", 0, "one"].identities.fieldloc == [(0, "outer"), (1, "one")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_23():
    assert recordarray2_d["outer", 0, "two"].identities.fieldloc == [(0, "outer"), (1, "two")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_24():
    assert recordarray2_d["outer", "one", 0].identities.fieldloc == [(0, "outer"), (1, "one")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_25():
    assert recordarray2_d["outer", "two", 0].identities.fieldloc == [(0, "outer"), (1, "two")]
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_26():
    assert recordarray2_d["outer", "one", 1, 0] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_27():
    assert recordarray2_d["outer", 1, "one", 0] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_28():
    assert recordarray2_d["outer", 1, 0, "one"] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_29():
    assert recordarray2_d[1, "outer", "one", 0] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_30():
    assert recordarray2_d[1, "outer", 0, "one"] == 2
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_31():
    assert recordarray2_d[1, 0, "outer", "one"] == 2

@pytest.mark.skip(reason="skip this for now")
def test_setidentities_32():
    with pytest.raises(ValueError) as excinfo:
        recordarray2_d["outer", 2, "two", 0, 99]
    assert str(excinfo.value) == 'in ListArray64 with identity [2, "outer", 0, "two"] attempting to get 99, index out of range'
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_33():
    assert recordarray2_d.identity == ()
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_34():
    assert recordarray2_d[2].identity == (2,)
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_35():
    assert recordarray2_d[2, "outer"].identity == (2, "outer")
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_36():
    assert recordarray2_d[2, "outer", 0].identity == (2, "outer", 0)
@pytest.mark.skip(reason="skip this for now")
def test_setidentities_37():
    assert recordarray2_d[2, "outer", 0, "two"].identity == (2, "outer", 0, "two")

def test_builder_tuple():
    builder = awkward1.layout.ArrayBuilder()
    assert str(builder.type) == 'unknown'
    assert awkward1.tolist(builder.snapshot()) == []

    builder.begintuple(0)
    builder.endtuple()

    builder.begintuple(0)
    builder.endtuple()

    builder.begintuple(0)
    builder.endtuple()

    assert str(builder.type) == '()'
    assert awkward1.tolist(builder.snapshot()) == [(), (), ()]

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

    assert str(builder.type) == '(bool, var * int64, float64)'
    assert awkward1.tolist(builder.snapshot()) == [(True, [1], 1.1), (False, [2, 2], 2.2), (True, [3, 3, 3], 3.3)]

def test_builder_record():
    builder = awkward1.layout.ArrayBuilder()
    assert str(builder.type) == 'unknown'
    assert awkward1.tolist(builder.snapshot()) == []

    builder.beginrecord()
    builder.endrecord()

    builder.beginrecord()
    builder.endrecord()

    builder.beginrecord()
    builder.endrecord()

    assert str(builder.type) == '{}'
    assert awkward1.tolist(builder.snapshot()) == [{}, {}, {}]

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

    assert str(builder.type) == '{"one": int64, "two": float64}'
    assert awkward1.tolist(builder.snapshot()) == [{"one": 1, "two": 1.1}, {"one": 2, "two": 2.2}, {"one": 3, "two": 3.3}]

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
        assert awkward1.tolist(awkward1.fromiter(datum)) == datum

def test_json():
    dataset = [
        '[{"one":1,"two":1.1},{"one":2,"two":2.2},{"one":3,"two":3.3}]',
        '[{"one":1,"two":[1.1,2.2,3.3]},{"one":2,"two":[]},{"one":3,"two":[4.4,5.5]}]',
        '[[{"one":1,"two":1.1},{"one":2,"two":2.2},{"one":3,"two":3.3}],[],[{"one":4,"two":4.4},{"one":5,"two":5.5}]]',
        '[{"one":{"x":1,"y":1},"two":1.1},{"one":{"x":2,"y":2},"two":2.2},{"one":{"x":3,"y":3},"two":3.3}]',
    ]
    for datum in dataset:
        assert awkward1.tojson(awkward1.fromjson(datum)) == datum
