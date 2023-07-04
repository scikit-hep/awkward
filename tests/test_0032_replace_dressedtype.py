# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_types_with_parameters():
    t = ak.types.UnknownType()
    assert t.parameters == {}

    with pytest.raises(TypeError):
        ak.types.UnknownType(parameters={"latitude": ["val", "ue"]})
    t = ak.types.UnknownType()
    assert t.parameters == {}

    t = ak.types.NumpyType("int32", parameters={"latitude": ["val", "ue"]})
    assert t.parameters == {"latitude": ["val", "ue"]}
    t = ak.types.NumpyType("float64", parameters={"latitude": ["val", "ue"]})
    assert t.parameters == {"latitude": ["val", "ue"]}
    t = ak.types.ArrayType(
        ak.types.NumpyType("int32", parameters={"latitude": ["val", "ue"]}), 100
    )
    assert t.content.parameters == {"latitude": ["val", "ue"]}
    t = ak.types.ListType(
        ak.types.NumpyType("int32"), parameters={"latitude": ["val", "ue"]}
    )
    assert t.parameters == {"latitude": ["val", "ue"]}
    t = ak.types.RegularType(
        ak.types.NumpyType("int32"), 5, parameters={"latitude": ["val", "ue"]}
    )
    assert t.parameters == {"latitude": ["val", "ue"]}
    t = ak.types.OptionType(
        ak.types.NumpyType("int32"), parameters={"latitude": ["val", "ue"]}
    )
    assert t.parameters == {"latitude": ["val", "ue"]}
    t = ak.types.UnionType(
        (ak.types.NumpyType("int32"), ak.types.NumpyType("float64")),
        parameters={"latitude": ["val", "ue"]},
    )
    assert t.parameters == {"latitude": ["val", "ue"]}
    t = ak.types.RecordType(
        [
            ak.types.NumpyType("int32"),
            ak.types.NumpyType("float64"),
        ],
        fields=["one", "two"],
        parameters={"latitude": ["val", "ue"]},
    )
    assert t.parameters == {"latitude": ["val", "ue"]}


def test_dress():
    class Dummy(ak.highlevel.Array):
        def __str__(self):
            return f"<Dummy {super().__str__()}>"

    ns = {"Dummy": Dummy}

    x = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6, 6])),
        size=3,
        parameters={"__list__": "Dummy"},
    )
    a = ak.highlevel.Array(x, behavior=ns, check_valid=True)
    assert str(a) == "<Dummy [[1.1, 2.2, 3.3], [4.4, 5.5, 6]]>"

    x2 = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 2, 2], dtype=np.int64)),
        x,
    )
    a2 = ak.highlevel.Array(x2, behavior=ns, check_valid=True)
    # columns limit changed from 40 to 80 in v2
    assert (
        repr(a2)
        == "<Array [<Dummy [[1.1, 2.2, 3.3], [4.4, 5.5, 6]]>, <Dummy []>, <Dummy []>] type='...'>"
    )
    assert str(a2[0]) == "<Dummy [[1.1, 2.2, 3.3], [4.4, 5.5, 6]]>"
    assert str(a2[1]) == "<Dummy []>"
    assert str(a2[2]) == "<Dummy []>"


def test_record_name():
    builder = ak.highlevel.ArrayBuilder()

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

    assert str(a.layout.form.type) == "Dummy[one: int64, two: float64]"
    assert a.layout.form.type.parameters == {"__record__": "Dummy"}


def test_builder_string():
    builder = ak.ArrayBuilder()
    builder.bytestring(b"one")
    builder.bytestring(b"two")
    builder.bytestring(b"three")

    a1 = builder.snapshot()
    assert str(a1) == "[b'one', b'two', b'three']"

    builder = ak.highlevel.ArrayBuilder()
    builder.bytestring(b"one")
    builder.bytestring(b"two")
    builder.bytestring(b"three")

    a = builder.snapshot()

    assert str(a) == "[b'one', b'two', b'three']"
    assert to_list(a) == [b"one", b"two", b"three"]
    assert (
        ak.operations.to_json(a, convert_bytes=bytes.decode) == '["one","two","three"]'
    )
    assert repr(a) == "<Array [b'one', b'two', b'three'] type='3 * bytes'>"
    assert str(ak.operations.type(a)) == "3 * bytes"

    builder = ak.highlevel.ArrayBuilder()

    builder.string("one")
    builder.string("two")
    builder.string("three")

    a = builder.snapshot()
    assert str(a) == "['one', 'two', 'three']"
    assert to_list(a) == ["one", "two", "three"]
    assert ak.operations.to_json(a) == '["one","two","three"]'
    assert repr(a) == "<Array ['one', 'two', 'three'] type='3 * string'>"
    assert str(ak.operations.type(a)) == "3 * string"

    builder = ak.highlevel.ArrayBuilder()

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
    assert ak.operations.to_json(a) == '[["one","two","three"],[],["four","five"]]'
    assert str(ak.operations.type(a)) == "3 * var * string"


def test_fromiter_fromjson():
    assert to_list(ak.from_iter(["one", "two", "three"])) == ["one", "two", "three"]
    assert to_list(ak.from_iter([["one", "two", "three"], [], ["four", "five"]])) == [
        ["one", "two", "three"],
        [],
        ["four", "five"],
    ]


def test_fromjson():
    assert to_list(ak.operations.from_json('["one", "two", "three"]')) == [
        "one",
        "two",
        "three",
    ]
    assert to_list(
        ak.operations.from_json('[["one", "two", "three"], [], ["four", "five"]]')
    ) == [["one", "two", "three"], [], ["four", "five"]]
