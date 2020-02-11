# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools
import collections

import pytest
import numpy

import awkward1
import awkward1.behaviors.string

py27 = (sys.version_info[0] < 3)

def test_fromnumpy():
    a = numpy.arange(2*3*5).reshape((2, 3, 5))
    b = awkward1.fromnumpy(a)
    assert awkward1.tolist(a) == awkward1.tolist(b)

def test_highlevel():
    a = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    assert repr(a) == "<Array [[1.1, 2.2, 3.3], ... [7.7, 8.8, 9.9]] type='5 * var * float64'>"
    assert str(a) == "[[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]"

    b = awkward1.Array(numpy.arange(100, dtype=numpy.int32))
    assert repr(b) == "<Array [0, 1, 2, 3, 4, ... 95, 96, 97, 98, 99] type='100 * int32'>"
    assert str(b) == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]"

    c = awkward1.Array('[{"one": 3.14, "two": [1.1, 2.2]}, {"one": 99.9, "two": [-3.1415926]}]')
    assert repr(c) == "<Array [{one: 3.14, two: [1.1, ... -3.14]}] type='2 * {\"one\": float64, \"two\": va...'>"
    assert str(c) == "[{one: 3.14, two: [1.1, 2.2]}, {one: 99.9, two: [-3.14]}]"

class Dummy(awkward1.highlevel.Array):
    pass

def test_string1():
    a = awkward1.Array(numpy.array([ord(x) for x in "hey there"], dtype=numpy.uint8))
    a.__class__ = awkward1.behaviors.string.CharBehavior
    assert str(a) == str(b"hey there")
    assert repr(a) == repr(b"hey there")

def test_string2():
    content = awkward1.layout.NumpyArray(numpy.array([ord(x) for x in "heythere"], dtype=numpy.uint8))
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 8])), content)
    a = awkward1.Array(listoffsetarray)

    assert isinstance(a, awkward1.Array)
    assert not isinstance(a, awkward1.behaviors.string.StringBehavior)
    assert awkward1.tolist(a) == [[104, 101, 121], [], [116, 104, 101, 114, 101]]

    assert repr(a.type) == "3 * var * uint8"
    assert repr(a[0].type) == "3 * uint8"
    assert repr(a[1].type) == "0 * uint8"
    assert repr(a[2].type) == "5 * uint8"

    assert repr(a) == "<Array [[104, 101, 121], ... 101, 114, 101]] type='3 * var * uint8'>"
    assert str(a) == "[[104, 101, 121], [], [116, 104, 101, 114, 101]]"
    assert repr(a[0]) == "<Array [104, 101, 121] type='3 * uint8'>"
    assert repr(a[1]) == "<Array [] type='0 * uint8'>"
    assert repr(a[2]) == "<Array [116, 104, 101, 114, 101] type='5 * uint8'>"

    a = awkward1.Array(listoffsetarray.astype(awkward1.string))
    assert isinstance(a, awkward1.Array)
    assert awkward1.tolist(a) == ['hey', '', 'there']

    assert repr(a.type) == "3 * string"
    assert repr(a[0].type) == "3 * utf8"
    assert repr(a[1].type) == "0 * utf8"
    assert repr(a[2].type) == "5 * utf8"

    if py27:
        assert repr(a) == "<Array [u'hey', u'', u'there'] type='3 * string'>"
        assert repr(a[0]) == "u'hey'"
        assert repr(a[1]) == "u''"
        assert repr(a[2]) == "u'there'"
    else:
        assert repr(a) == "<Array ['hey', '', 'there'] type='3 * string'>"
        assert repr(a[0]) == "'hey'"
        assert repr(a[1]) == "''"
        assert repr(a[2]) == "'there'"

def test_accepts():
    dressed1 = awkward1.type.ListType(awkward1.type.PrimitiveType("float64"), {"__record__": "Dummy"})
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.float64))
    listoffsetarray = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5])), content).astype(dressed1)

    dressed2 = awkward1.type.PrimitiveType("float64", {"__record__": "Dummy"})
    with pytest.raises(ValueError):
        awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5])), content).astype(dressed2)

class D(awkward1.highlevel.Array):
    @staticmethod
    def typestr(baretype, parameters):
        return "D[{0}]".format(baretype)

def test_type_propagation():
    array = awkward1.Array([[{"one": 1, "two": [1.0, 1.1]}, {"one": 2, "two": [2.0]}, {"one": 3, "two": [3.0, 3.1, 3.2]}], [], [{"one": 4, "two": []}, {"one": 5, "two": [5.0, 5.1, 5.2, 5.3]}]])
    assert awkward1.tolist(array) == [[{"one": 1, "two": [1.0, 1.1]}, {"one": 2, "two": [2.0]}, {"one": 3, "two": [3.0, 3.1, 3.2]}], [], [{"one": 4, "two": []}, {"one": 5, "two": [5.0, 5.1, 5.2, 5.3]}]]
    assert repr(array.type) in ('3 * var * {"one": int64, "two": var * float64}', '3 * var * {"two": var * float64, "one": int64}')

    dfloat64 = awkward1.type.PrimitiveType("float64", {"__record__": "D", "__typestr__": "D[float64]"})
    dvarfloat64 = awkward1.type.ListType(dfloat64, {"__record__": "D", "__typestr__": "D[var * D[float64]]"})
    dint64 = awkward1.type.PrimitiveType("int64", {"__record__": "D", "__typestr__": "D[int64]"})
    drec = awkward1.type.RecordType(collections.OrderedDict([("one", dint64), ("two", dvarfloat64)]), {"__record__": "D", "__typestr__": "D[{\"one\": D[int64], \"two\": D[var * D[float64]]}]"})
    dvarrec = awkward1.type.ListType(drec, {"__record__": "D", "__typestr__": "D[var * D[{\"one\": D[int64], \"two\": D[var * D[float64]]}]]"})

    array = awkward1.Array(array.layout.astype(dvarrec))

    assert array.layout.type == dvarrec
    assert array.layout.content.type == drec
    assert array.layout.content.field("one").type == dint64
    assert array.layout.content.field("two").type == dvarfloat64
    assert array.layout.content.field("two").content.type == dfloat64

    assert array.layout[-1].type == drec
    assert array.layout[-1]["one"].type == dint64
    assert array.layout[-1]["two"].type == dvarfloat64
    assert array.layout[-1]["two"][1].type == dfloat64
    assert array.layout[-1, "one"].type == dint64
    assert array.layout[-1, "two"].type == dvarfloat64
    assert array.layout[-1, "two", 1].type == dfloat64
    assert array.layout["one", -1].type == dint64
    assert array.layout["two", -1].type == dvarfloat64
    assert array.layout["two", -1, 1].type == dfloat64

    assert array.layout[1:].type == dvarrec
    assert array.layout[1:, "one"].type == awkward1.type.ListType(dint64)
    assert array.layout["one", 1:].type == awkward1.type.ListType(dint64)

    assert array.layout[[2, 1]].type == dvarrec
    assert array.layout[[2, 1], "one"].type == awkward1.type.ListType(dint64)

    array2 = awkward1.layout.NumpyArray(numpy.arange(2*3*5, dtype=numpy.int64).reshape(2, 3, 5)).astype(awkward1.type.RegularType(awkward1.type.RegularType(dint64, 5), 3))

    assert repr(array2.type) == "3 * 5 * D[int64]"
    assert repr(array2[0].type) == "5 * D[int64]"
    assert repr(array2[0, 0].type) == "D[int64]"
    assert array2[-1, -1, -1] == 29
