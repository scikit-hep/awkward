# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_array():
    array = ak.Array(["this", {"x": ["is", 1, 2, None]}])
    assert ak.type(array) == array.type
    assert isinstance(array.type, ak.types.ArrayType)


def test_record():
    record = ak.Record({"y": ["this", {"x": ["is", 1, 2, None]}]})
    assert ak.type(record) == record.type
    assert isinstance(record.type, ak.types.ScalarType)


def test_none():
    assert ak.type(None) == ak.types.ScalarType(ak.types.UnknownType())


def test_unknown():
    with pytest.raises(TypeError):
        ak.type(object())


def test_bare_string():
    assert ak.type("hello") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "char"}, typestr="char"), 5
    )
    assert ak.type(b"hello") == ak.types.ArrayType(
        ak.types.NumpyType("uint8", parameters={"__array__": "byte"}, typestr="byte"), 5
    )


def test_array_string():
    assert ak.type(["hello"]) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType(
                "uint8", parameters={"__array__": "char"}, typestr="char"
            ),
            parameters={"__array__": "string"},
            typestr="string",
        ),
        1,
    )
    assert ak.type([b"hello"]) == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType(
                "uint8", parameters={"__array__": "byte"}, typestr="byte"
            ),
            parameters={"__array__": "bytestring"},
            typestr="bytes",
        ),
        1,
    )
