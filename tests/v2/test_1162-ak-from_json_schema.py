# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_boolean():
    result = ak._v2.operations.convert.from_json_schema(
        " [ true ,false, true, true, false]    ",
        {"type": "array", "items": {"type": "boolean"}},
    )
    assert result.tolist() == [True, False, True, True, False]

    result = ak._v2.operations.convert.from_json_schema(
        "[]",
        {"type": "array", "items": {"type": "boolean"}},
    )
    assert result.tolist() == []


def test_integer():
    result = ak._v2.operations.convert.from_json_schema(
        " [ 1 ,2 ,3.0, 4, 5]  \n  ",
        {"type": "array", "items": {"type": "integer"}},
    )
    assert result.tolist() == [1, 2, 3, 4, 5]

    result = ak._v2.operations.convert.from_json_schema(
        "[ ]",
        {"type": "array", "items": {"type": "integer"}},
    )
    assert result.tolist() == []


def test_number():
    result = ak._v2.operations.convert.from_json_schema(
        " [ 1 ,2,3.14, 4, 5]",
        {"type": "array", "items": {"type": "number"}},
    )
    assert result.tolist() == [1, 2, 3.14, 4, 5]


def test_option_boolean():
    result = ak._v2.operations.convert.from_json_schema(
        " [ true ,false ,null , true, false]",
        {"type": "array", "items": {"type": ["boolean", "null"]}},
    )
    assert result.tolist() == [True, False, None, True, False]


def test_option_integer():
    result = ak._v2.operations.convert.from_json_schema(
        " [ 1 ,2,null,4, 5]",
        {"type": "array", "items": {"type": ["null", "integer"]}},
    )
    assert result.tolist() == [1, 2, None, 4, 5]

    result = ak._v2.operations.convert.from_json_schema(
        " [ ]",
        {"type": "array", "items": {"type": ["null", "integer"]}},
    )
    assert result.tolist() == []


def test_string():
    result = ak._v2.operations.convert.from_json_schema(
        r' [ "" ,"two","three \u2192 3", "\"four\"", "fi\nve"]',
        {"type": "array", "items": {"type": "string"}},
    )
    assert result.tolist() == ["", "two", "three \u2192 3", '"four"', "fi\nve"]

    result = ak._v2.operations.convert.from_json_schema(
        r"[]",
        {"type": "array", "items": {"type": "string"}},
    )
    assert result.tolist() == []


def test_option_string():
    result = ak._v2.operations.convert.from_json_schema(
        r' [ "" ,null ,"three \u2192 3", "\"four\"", "fi\nve"]',
        {"type": "array", "items": {"type": ["null", "string"]}},
    )
    assert result.tolist() == ["", None, "three \u2192 3", '"four"', "fi\nve"]

    result = ak._v2.operations.convert.from_json_schema(
        r"[]",
        {"type": "array", "items": {"type": ["null", "string"]}},
    )
    assert result.tolist() == []


def test_array_integer():
    result = ak._v2.operations.convert.from_json_schema(
        " [ [ 1 ,2, 3], [], [4, 5]]",
        {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
    )
    assert result.tolist() == [[1, 2, 3], [], [4, 5]]
    assert str(result.type) == "3 * var * int64"

    result = ak._v2.operations.convert.from_json_schema(
        "[]",
        {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
    )
    assert result.tolist() == []


def test_regulararray_integer():
    result = ak._v2.operations.convert.from_json_schema(
        "[[1, 2, 3], [4, 5, 6]]",
        {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
    )
    assert result.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert str(result.type) == "2 * 3 * int64"


def test_option_regulararray_integer():
    result = ak._v2.operations.convert.from_json_schema(
        "[[1, 2, 3], null, [4, 5, 6]]",
        {
            "type": "array",
            "items": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
    )
    assert result.tolist() == [[1, 2, 3], None, [4, 5, 6]]
    assert str(result.type) == "3 * option[3 * int64]"


def test_option_array_integer():
    result = ak._v2.operations.convert.from_json_schema(
        " [ [ 1 ,2,3 ],null,[ ], [4, 5]]",
        {
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
    )
    assert result.tolist() == [[1, 2, 3], None, [], [4, 5]]
    assert str(result.type) == "4 * option[var * int64]"

    result = ak._v2.operations.convert.from_json_schema(
        "[]",
        {
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
    )
    assert result.tolist() == []


def test_option_array_option_integer():
    result = ak._v2.operations.convert.from_json_schema(
        " [ [ 1 ,2,3 ],null,[ ] ,[null, 5]]",
        {
            "type": "array",
            "items": {
                "type": ["null", "array"],
                "items": {"type": ["integer", "null"]},
            },
        },
    )
    assert result.tolist() == [[1, 2, 3], None, [], [None, 5]]

    result = ak._v2.operations.convert.from_json_schema(
        "[]",
        {
            "type": "array",
            "items": {
                "type": ["null", "array"],
                "items": {"type": ["integer", "null"]},
            },
        },
    )
    assert result.tolist() == []


def test_array_array_integer():
    result = ak._v2.operations.convert.from_json_schema(
        " [ [ [ 1 ,2,3 ] ] ,[ [], [4, 5]], []]",
        {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
    )
    assert result.tolist() == [[[1, 2, 3]], [[], [4, 5]], []]


def test_record():
    result = ak._v2.operations.convert.from_json_schema(
        ' [ { "x" :1 ,"y":1.1},{"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]',
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.tolist() == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]

    result = ak._v2.operations.convert.from_json_schema(
        "[]",
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.tolist() == []


def test_option_record():
    result = ak._v2.operations.convert.from_json_schema(
        ' [ { "x" : 1 ,"y":1.1},null ,{"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]',
        {
            "type": "array",
            "items": {
                "type": ["object", "null"],
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.tolist() == [
        {"x": 1, "y": 1.1},
        None,
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]

    result = ak._v2.operations.convert.from_json_schema(
        "[]",
        {
            "type": "array",
            "items": {
                "type": ["object", "null"],
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.tolist() == []
