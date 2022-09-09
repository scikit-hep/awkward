# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_types_with_parameters():
    t = ak._v2.types.UnknownType()
    assert t.parameters == {}

    t = ak._v2.types.UnknownType(parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}

    t = ak._v2.types.NumpyType("int32", parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.NumpyType("float64", parameters={"__array__": ["val", "ue"]})
    assert t.parameters == {"__array__": ["val", "ue"]}
    t = ak._v2.types.ArrayType(
        ak._v2.types.NumpyType("int32", parameters={"__array__": ["val", "ue"]}), 100
    )
    assert t.content.parameters == {"__array__": ["val", "ue"]}
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
        [
            ak._v2.types.NumpyType("int32"),
            ak._v2.types.NumpyType("float64"),
        ],
        fields=["one", "two"],
        parameters={"__array__": ["val", "ue"]},
    )
    assert t.parameters == {"__array__": ["val", "ue"]}

    t = ak._v2.types.UnknownType(
        parameters={"key1": ["val", "ue"], "__record__": "one \u2192 two"}
    )
    assert t.parameters == {"__record__": "one \u2192 two", "key1": ["val", "ue"]}

    assert t == ak._v2.types.UnknownType(
        parameters={"__record__": "one \u2192 two", "key1": ["val", "ue"]}
    )
    assert t != ak._v2.types.UnknownType(parameters={"__array__": ["val", "ue"]})


def test_dress():
    class Dummy(ak._v2.highlevel.Array):
        def __repr__(self):
            return f"<Dummy {str(self)}>"

    ns = {"Dummy": Dummy}

    x = ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    x.parameters["__array__"] = "Dummy"
    a = ak._v2.highlevel.Array(x, behavior=ns, check_valid=True)
    assert repr(a) == "<Dummy [1.1, 2.2, 3.3, 4.4, 5.5]>"

    x2 = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64)),
        ak._v2.contents.NumpyArray(
            np.array([1.1, 2.2, 3.3, 4.4, 5.5]), parameters={"__array__": "Dummy"}
        ),
    )
    a2 = ak._v2.highlevel.Array(x2, behavior=ns, check_valid=True)
    # columns limit changed from 40 to 80 in v2
    assert (
        repr(a2)
        == "<Array [<Dummy [1.1, 2.2, 3.3]>, <Dummy []>, <Dummy [4.4, 5.5]>] type='3 * ...'>"
    )
    assert repr(a2[0]) == "<Dummy [1.1, 2.2, 3.3]>"
    assert repr(a2[1]) == "<Dummy []>"
    assert repr(a2[2]) == "<Dummy [4.4, 5.5]>"


def test_record_name():
    typestrs = {}
    builder = ak._v2.highlevel.ArrayBuilder()

    builder.begin_record("Dummy")
    builder.field("one")
    builder.integer(1)
    builder.field("two")
    builder.real(1.1)
    builder.end_record()

    builder.begin_record("Dummy")
    builder.field("two")
    builder.real(2.2)
    builder.field("one")
    builder.integer(2)
    builder.end_record()

    a = builder.snapshot()

    assert str(a.layout.form._type(typestrs)) == "Dummy[one: int64, two: float64]"
    assert a.layout.form._type(typestrs).parameters == {"__record__": "Dummy"}


def test_builder_string():
    builder = ak.ArrayBuilder()
    builder.bytestring(b"one")
    builder.bytestring(b"two")
    builder.bytestring(b"three")

    a1 = builder.snapshot()
    assert str(a1) == "[b'one', b'two', b'three']"

    builder = ak._v2.highlevel.ArrayBuilder()
    builder.bytestring(b"one")
    builder.bytestring(b"two")
    builder.bytestring(b"three")

    a = builder.snapshot()

    assert str(a) == "[b'one', b'two', b'three']"
    assert to_list(a) == [b"one", b"two", b"three"]
    assert (
        ak._v2.operations.to_json(a, convert_bytes=bytes.decode)
        == '["one","two","three"]'
    )
    assert repr(a) == "<Array [b'one', b'two', b'three'] type='3 * bytes'>"
    assert str(ak._v2.operations.type(a)) == "3 * bytes"

    builder = ak._v2.highlevel.ArrayBuilder()

    builder.string("one")
    builder.string("two")
    builder.string("three")

    a = builder.snapshot()
    assert str(a) == "['one', 'two', 'three']"
    assert to_list(a) == ["one", "two", "three"]
    assert ak._v2.operations.to_json(a) == '["one","two","three"]'
    assert repr(a) == "<Array ['one', 'two', 'three'] type='3 * string'>"
    assert str(ak._v2.operations.type(a)) == "3 * string"

    builder = ak._v2.highlevel.ArrayBuilder()

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
    assert to_list(a) == [["one", "two", "three"], [], ["four", "five"]]
    assert ak._v2.operations.to_json(a) == '[["one","two","three"],[],["four","five"]]'
    assert str(ak._v2.operations.type(a)) == "3 * var * string"


def test_fromiter_fromjson():
    assert to_list(ak._v2.from_iter(["one", "two", "three"])) == ["one", "two", "three"]
    assert to_list(
        ak._v2.from_iter([["one", "two", "three"], [], ["four", "five"]])
    ) == [["one", "two", "three"], [], ["four", "five"]]


def test_fromjson():
    assert to_list(ak._v2.operations.from_json('["one", "two", "three"]')) == [
        "one",
        "two",
        "three",
    ]
    assert to_list(
        ak._v2.operations.from_json('[["one", "two", "three"], [], ["four", "five"]]')
    ) == [["one", "two", "three"], [], ["four", "five"]]
