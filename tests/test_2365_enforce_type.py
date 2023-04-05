# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_simple():
    array = (
        ak.forms.from_dict(
            {
                "class": "IndexedArray",
                "index": "i64",
                "content": {
                    "class": "RecordArray",
                    "fields": ["x"],
                    "contents": ["int64"],
                },
            }
        )
        .length_zero_array()
        .x
    )

    type = ak.types.from_datashape("int8", highlevel=False)

    result = ak.enforce_type(type, array)
    assert str(result.type.content) == str(type)


def test_reg():
    array = ak.Array([[1, 2, 3], [4, 5, 6]])

    type = ak.types.from_datashape("3 * int64", highlevel=False)
    assert not isinstance(type, ak.types.ArrayType)

    result = ak.enforce_type(type, array)
    assert str(result.type.content) == str(type)


def test_var():
    array = ak.to_regular(ak.Array([[1, 2, 3.0], [4, 5, 6]]))

    type = ak.types.from_datashape("var * int64", highlevel=False)
    assert not isinstance(type, ak.types.ArrayType)

    result = ak.enforce_type(type, array)
    assert str(result.type.content) == str(type)


def test_record():
    array = ak.zip({"x": [[1, 2, 3]], "y": [[1.0, 2.0, 3.0]]})

    type = ak.types.from_datashape("3 * {x: int64,y: int64}", highlevel=False)
    assert not isinstance(type, ak.types.ArrayType)

    result = ak.enforce_type(type, array)
    assert str(result.type.content) == str(type)


def test_option():
    array = ak.Array([[1, 2, 3, 4], [5, 6, 7]])

    type = ak.types.from_datashape("var * ?float32", highlevel=False)
    assert not isinstance(type, ak.types.ArrayType)

    result = ak.enforce_type(type, array)
    assert str(result.type.content) == str(type)
