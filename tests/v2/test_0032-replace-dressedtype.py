# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.convert.to_list


@pytest.mark.skip(reason="FIXME: Fix params for types")
def test_types_with_parameters():
    t = ak._v2.types.UnknownType()
    assert t.parameters == None
    t.parameters["__array__"] = ["val", "ue"]
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.UnknownType(parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}

    t = ak._v2.types.NumpyType("int32", parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.NumpyType("float64", parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.ArrayType(
        ak._v2.types.NumpyType("int32"), 100, parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.ListType(
        ak._v2.types.NumpyType("int32"), parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.RegularType(
        ak._v2.types.NumpyType("int32"), 5, parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.OptionType(
        ak._v2.types.NumpyType("int32"), parameters={"__array__": ["val", "ue"]}
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.UnionType(
        (ak._v2.types.NumpyType("int32"), ak._v2.types.NumpyType("float64")),
        parameters={"__array__": ["val", "ue"]},
    )
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.RecordType(
        {
            "one": ak._v2.types.NumpyType("int32"),
            "two": ak._v2.types.NumpyType("float64"),
        },
        parameters={"__array__": ["val", "ue"]},
    )
    assert t.parameters == {"__array__": ["val", "ue"]}

    t = ak._v2.types.UnknownType(
        parameters={"key1": ["val", "ue"], "__record__": u"one \u2192 two"}
    )
    assert t.parameters == {"__record__": u"one \u2192 two", "key1": ["val", "ue"]}

    assert t == ak._v2.types.UnknownType(
        parameters={"__record__": u"one \u2192 two", "key1": ["val", "ue"]}
    )
    assert t != ak._v2.types.UnknownType(parameters={"__array__": ["val", "ue"]})


@pytest.mark.skip(reason="FIXME: Fix params for types")
def test_dress():
    class Dummy(ak.highlevel.Array):
        def __repr__(self):
            return "<Dummy {0}>".format(str(self))

    ns = {"Dummy": Dummy}

    x = ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    x.setparameter("__array__", "Dummy")
    a = ak.Array(x, behavior=ns, check_valid=True)
    assert repr(a) == "<Dummy [1.1, 2.2, 3.3, 4.4, 5.5]>"

    x2 = ak._v2.contents.ListOffsetArray64(
        ak._v2.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64)),
        ak._v2.contents.NumpyArray(
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


@pytest.mark.skip(reason="FIXME: Fix params for types")
def test_record_name():
    typestrs = {}
    builder = ak._v2.contents.ArrayBuilder()

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
    assert repr(a.type(typestrs)) == 'Dummy["one": int64, "two": float64]'
    assert a.type(typestrs).parameters == {"__record__": "Dummy"}


def test_builder_string():
    builder = ak.ArrayBuilder()

    builder.bytestring(b"one")
    builder.bytestring(b"two")
    builder.bytestring(b"three")

    a = builder.snapshot()
    assert str(a) == "[b'one', b'two', b'three']"
    assert ak.to_list(a) == [b"one", b"two", b"three"]
    assert ak.to_json(a) == '["one","two","three"]'
    assert repr(a) == "<Array [b'one', b'two', b'three'] type='3 * bytes'>"
    assert repr(ak.type(a)) == "3 * bytes"

    builder = ak.ArrayBuilder()

    builder.string("one")
    builder.string("two")
    builder.string("three")

    a = builder.snapshot()
    assert str(a) == "['one', 'two', 'three']"
    assert ak.to_list(a) == ["one", "two", "three"]
    assert ak.to_json(a) == '["one","two","three"]'
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
