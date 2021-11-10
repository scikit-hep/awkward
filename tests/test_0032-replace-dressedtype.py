# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_types_with_parameters():
    t = ak.types.UnknownType()
    assert t.parameters == {}
    t.parameters = {"__array__": ["val", "ue"]}
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.UnknownType(parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}

    t = ak.types.PrimitiveType("int32", parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.PrimitiveType("float64", parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.ArrayType(
        ak.types.PrimitiveType("int32"), 100, parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.ListType(
        ak.types.PrimitiveType("int32"), parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.RegularType(
        ak.types.PrimitiveType("int32"), 5, parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.OptionType(
        ak.types.PrimitiveType("int32"), parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.UnionType(
        (ak.types.PrimitiveType("int32"), ak.types.PrimitiveType("float64")),
        parameters={"__array__": ["val", "ue"]},
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak.types.RecordType(
        {
            "one": ak.types.PrimitiveType("int32"),
            "two": ak.types.PrimitiveType("float64"),
        },
        parameters={"__array__": ["val", "ue"]},
    )
    assert t.parameters == {"__array__": ["val", "ue"]}

    t = ak.types.UnknownType(
        parameters={"key1": ["val", "ue"], "__record__": u"one \u2192 two"}
    )
    assert t.parameters == {"__record__": u"one \u2192 two", "key1": ["val", "ue"]}

    assert t == ak.types.UnknownType(
        parameters={"__record__": u"one \u2192 two", "key1": ["val", "ue"]}
    )
    assert t != ak.types.UnknownType(parameters={"__array__": ["val", "ue"]})


def test_dress():
    class Dummy(ak.highlevel.Array):
        def __repr__(self):
            return "<Dummy {0}>".format(str(self))

    ns = {"Dummy": Dummy}

    x = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    x.setparameter("__array__", "Dummy")
    a = ak.Array(x, behavior=ns, check_valid=True)
    assert repr(a) == "<Dummy [1.1, 2.2, 3.3, 4.4, 5.5]>"

    x2 = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64)),
        ak.layout.NumpyArray(
            np.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"__array__": "Dummy"}
        ),
    )
    a2 = ak.Array(x2, behavior=ns, check_valid=True)
    assert (
        repr(a2)
        == "<Array [<Dummy [1.1, 2.2, 3.3]>, ... ] type='3 * var * float64[parameters={\"__ar...'>"
    )
    assert repr(a2[0]) == "<Dummy [1.1, 2.2, 3.3]>"
    assert repr(a2[1]) == "<Dummy []>"
    assert repr(a2[2]) == "<Dummy [4.4, 5.5]>"


def test_record_name():
    typestrs = {}
    builder = ak.layout.ArrayBuilder()

    builder.beginrecord("Dummy")
    builder.field("one")
    builder.integer(1)
    builder.field("two")
    builder.real(1.1)
    builder.endrecord()

    builder.beginrecord("Dummy")
    builder.field("two")
    builder.real(2.2)
    builder.field("one")
    builder.integer(2)
    builder.endrecord()

    a = builder.snapshot()
    if not (ak._util.py27 or ak._util.py35):
        assert repr(a.type(typestrs)) == 'Dummy["one": int64, "two": float64]'
    assert a.type(typestrs).parameters == {"__record__": "Dummy"}


def test_builder_string():
    builder = ak.ArrayBuilder()

    builder.bytestring(b"one")
    builder.bytestring(b"two")
    builder.bytestring(b"three")

    a = builder.snapshot()
    if ak._util.py27:
        assert str(a) == "['one', 'two', 'three']"
    else:
        assert str(a) == "[b'one', b'two', b'three']"
    assert ak.to_list(a) == [b"one", b"two", b"three"]
    assert ak.to_json(a) == '["one","two","three"]'
    if ak._util.py27:
        assert repr(a) == "<Array ['one', 'two', 'three'] type='3 * bytes'>"
    else:
        assert repr(a) == "<Array [b'one', b'two', b'three'] type='3 * bytes'>"
    assert repr(ak.type(a)) == "3 * bytes"

    builder = ak.ArrayBuilder()

    builder.string("one")
    builder.string("two")
    builder.string("three")

    a = builder.snapshot()
    if ak._util.py27:
        assert str(a) == "[u'one', u'two', u'three']"
    else:
        assert str(a) == "['one', 'two', 'three']"
    assert ak.to_list(a) == ["one", "two", "three"]
    assert ak.to_json(a) == '["one","two","three"]'
    if ak._util.py27:
        assert repr(a) == "<Array [u'one', u'two', u'three'] type='3 * string'>"
    else:
        assert repr(a) == "<Array ['one', 'two', 'three'] type='3 * string'>"
    assert repr(ak.type(a)) == "3 * string"

    builder = ak.ArrayBuilder()

    builder.begin_list()
    builder.string("one")
    builder.string("two")
    builder.string("three")
    builder.end_list()

    builder.begin_list()
    builder.end_list()

    builder.begin_list()
    builder.string("four")
    builder.string("five")
    builder.end_list()

    a = builder.snapshot()
    if ak._util.py27:
        assert str(a) == "[[u'one', u'two', u'three'], [], [u'four', u'five']]"
    else:
        assert str(a) == "[['one', 'two', 'three'], [], ['four', 'five']]"
    assert ak.to_list(a) == [["one", "two", "three"], [], ["four", "five"]]
    assert ak.to_json(a) == '[["one","two","three"],[],["four","five"]]'
    assert repr(ak.type(a)) == "3 * var * string"


def test_fromiter_fromjson():
    assert ak.to_list(ak.from_iter(["one", "two", "three"])) == ["one", "two", "three"]
    assert ak.to_list(
        ak.from_iter([["one", "two", "three"], [], ["four", "five"]])
    ) == [["one", "two", "three"], [], ["four", "five"]]

    assert ak.to_list(ak.from_json('["one", "two", "three"]')) == [
        "one",
        "two",
        "three",
    ]
    assert ak.to_list(
        ak.from_json('[["one", "two", "three"], [], ["four", "five"]]')
    ) == [["one", "two", "three"], [], ["four", "five"]]
