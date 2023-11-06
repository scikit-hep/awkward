# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import awkward as ak

DIR = os.path.dirname(__file__)
SAMPLES_DIR = os.path.join(os.path.abspath(DIR), "samples")


def test_boolean():
    result = ak.operations.from_json(
        " [ true ,false, true, true, false]    ",
        schema={"type": "array", "items": {"type": "boolean"}},
    )
    assert result.to_list() == [True, False, True, True, False]
    assert str(result.type) == "5 * bool"

    result = ak.operations.from_json(
        " [ true ,false, true, true, false]    " * 2,
        schema={"type": "array", "items": {"type": "boolean"}},
        line_delimited=True,
    )
    assert result.to_list() == [True, False, True, True, False] * 2
    assert str(result.type) == "10 * bool"

    result = ak.operations.from_json(
        "",
        schema={"type": "array", "items": {"type": "boolean"}},
        line_delimited=True,
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * bool"

    result = ak.operations.from_json(
        "[]",
        schema={"type": "array", "items": {"type": "boolean"}},
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * bool"


def test_integer():
    with pytest.raises(ValueError):
        ak.operations.from_json(
            " [ 1 ,2 ,3.0, 4, 5]  \n  ",
            schema={"type": "array", "items": {"type": "integer"}},
        )

    result = ak.operations.from_json(
        " [ 1 ,2 ,3, 4, 5]  \n  ",
        schema={"type": "array", "items": {"type": "integer"}},
    )
    assert result.to_list() == [1, 2, 3, 4, 5]
    assert str(result.type) == "5 * int64"

    result = ak.operations.from_json(
        " [ 1 ,2 ,3, 4, 5]  \n  " * 2,
        schema={"type": "array", "items": {"type": "integer"}},
        line_delimited=True,
    )
    assert result.to_list() == [1, 2, 3, 4, 5] * 2
    assert str(result.type) == "10 * int64"

    result = ak.operations.from_json(
        "[ ]",
        schema={"type": "array", "items": {"type": "integer"}},
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * int64"


def test_number():
    result = ak.operations.from_json(
        " [ 1 ,2,3.14, 4, 5]",
        schema={"type": "array", "items": {"type": "number"}},
    )
    assert result.to_list() == [1, 2, 3.14, 4, 5]
    assert str(result.type) == "5 * float64"

    result = ak.operations.from_json(
        " [ 1 ,2,3.14, 4, 5]" * 2,
        schema={"type": "array", "items": {"type": "number"}},
        line_delimited=True,
    )
    assert result.to_list() == [1, 2, 3.14, 4, 5] * 2
    assert str(result.type) == "10 * float64"

    result = ak.operations.from_json(
        " [ ]",
        schema={"type": "array", "items": {"type": "number"}},
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * float64"


def test_option_boolean():
    result = ak.operations.from_json(
        " [ true ,false ,null , true, false]",
        schema={"type": "array", "items": {"type": ["boolean", "null"]}},
    )
    assert result.to_list() == [True, False, None, True, False]
    assert str(result.type) == "5 * ?bool"

    result = ak.operations.from_json(
        " [ true ,false ,null , true, false]" * 2,
        schema={"type": "array", "items": {"type": ["boolean", "null"]}},
        line_delimited=True,
    )
    assert result.to_list() == [True, False, None, True, False] * 2
    assert str(result.type) == "10 * ?bool"

    result = ak.operations.from_json(
        " [ ]",
        schema={"type": "array", "items": {"type": ["boolean", "null"]}},
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * ?bool"


def test_option_integer():
    result = ak.operations.from_json(
        " [ 1 ,2,null,4, 5]",
        schema={"type": "array", "items": {"type": ["null", "integer"]}},
    )
    assert result.to_list() == [1, 2, None, 4, 5]
    assert str(result.type) == "5 * ?int64"

    result = ak.operations.from_json(
        " [ 1 ,2,null,4, 5]" * 2,
        schema={"type": "array", "items": {"type": ["null", "integer"]}},
        line_delimited=True,
    )
    assert result.to_list() == [1, 2, None, 4, 5] * 2
    assert str(result.type) == "10 * ?int64"

    result = ak.operations.from_json(
        " [ ]",
        schema={"type": "array", "items": {"type": ["null", "integer"]}},
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * ?int64"


def test_string():
    result = ak.operations.from_json(
        r' [ "" ,"two","three \u2192 3", "\"four\"", "fi\nve"]',
        schema={"type": "array", "items": {"type": "string"}},
    )
    assert result.to_list() == ["", "two", "three \u2192 3", '"four"', "fi\nve"]
    assert str(result.type) == "5 * string"

    result = ak.operations.from_json(
        r' [ "" ,"two","three \u2192 3", "\"four\"", "fi\nve"]' * 2,
        schema={"type": "array", "items": {"type": "string"}},
        line_delimited=True,
    )
    assert result.to_list() == ["", "two", "three \u2192 3", '"four"', "fi\nve"] * 2
    assert str(result.type) == "10 * string"

    result = ak.operations.from_json(
        r"[]",
        schema={"type": "array", "items": {"type": "string"}},
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * string"


def test_option_string():
    result = ak.operations.from_json(
        r' [ "" ,null ,"three \u2192 3", "\"four\"", "fi\nve"]',
        schema={"type": "array", "items": {"type": ["null", "string"]}},
    )
    assert result.to_list() == ["", None, "three \u2192 3", '"four"', "fi\nve"]
    assert str(result.type) == "5 * ?string"

    result = ak.operations.from_json(
        r' [ "" ,null ,"three \u2192 3", "\"four\"", "fi\nve"]' * 2,
        schema={"type": "array", "items": {"type": ["null", "string"]}},
        line_delimited=True,
    )
    assert result.to_list() == ["", None, "three \u2192 3", '"four"', "fi\nve"] * 2
    assert str(result.type) == "10 * ?string"

    result = ak.operations.from_json(
        r"[]",
        schema={"type": "array", "items": {"type": ["null", "string"]}},
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * ?string"


def test_enum_string():
    result = ak.operations.from_json(
        r'["three", "two", "one", "one", "two", "three"]',
        schema={
            "type": "array",
            "items": {"type": "string", "enum": ["one", "two", "three"]},
        },
    )
    assert result.to_list() == ["three", "two", "one", "one", "two", "three"]
    assert isinstance(result.layout, ak.contents.IndexedArray)
    assert result.layout.index.data.tolist() == [2, 1, 0, 0, 1, 2]
    assert str(result.type) == "6 * categorical[type=string]"

    result = ak.operations.from_json(
        r'["three", "two", "one", "one", "two", "three"]' * 2,
        schema={
            "type": "array",
            "items": {"type": "string", "enum": ["one", "two", "three"]},
        },
        line_delimited=True,
    )
    assert result.to_list() == ["three", "two", "one", "one", "two", "three"] * 2
    assert isinstance(result.layout, ak.contents.IndexedArray)
    assert result.layout.index.data.tolist() == [2, 1, 0, 0, 1, 2] * 2
    assert str(result.type) == "12 * categorical[type=string]"

    result = ak.operations.from_json(
        r"[]",
        schema={
            "type": "array",
            "items": {"type": "string", "enum": ["one", "two", "three"]},
        },
    )
    assert result.to_list() == []
    assert isinstance(result.layout, ak.contents.IndexedArray)
    assert result.layout.index.data.tolist() == []
    assert str(result.type) == "0 * categorical[type=string]"


def test_option_enum_string():
    result = ak.operations.from_json(
        r'["three", "two", null, "one", "one", "two", "three"]',
        schema={
            "type": "array",
            "items": {"type": ["null", "string"], "enum": ["one", "two", "three"]},
        },
    )
    assert result.to_list() == ["three", "two", None, "one", "one", "two", "three"]
    assert isinstance(result.layout, ak.contents.IndexedOptionArray)
    assert result.layout.index.data.tolist() == [2, 1, -1, 0, 0, 1, 2]
    assert str(result.type) == "7 * ?categorical[type=string]"

    result = ak.operations.from_json(
        r'["three", "two", null, "one", "one", "two", "three"]' * 2,
        schema={
            "type": "array",
            "items": {"type": ["null", "string"], "enum": ["one", "two", "three"]},
        },
        line_delimited=True,
    )
    assert result.to_list() == ["three", "two", None, "one", "one", "two", "three"] * 2
    assert isinstance(result.layout, ak.contents.IndexedOptionArray)
    assert result.layout.index.data.tolist() == [2, 1, -1, 0, 0, 1, 2] * 2
    assert str(result.type) == "14 * ?categorical[type=string]"

    result = ak.operations.from_json(
        r"[]",
        schema={
            "type": "array",
            "items": {"type": ["null", "string"], "enum": ["one", "two", "three"]},
        },
    )
    assert result.to_list() == []
    assert isinstance(result.layout, ak.contents.IndexedOptionArray)
    assert result.layout.index.data.tolist() == []
    assert str(result.type) == "0 * ?categorical[type=string]"


def test_array_integer():
    result = ak.operations.from_json(
        " [ [ 1 ,2, 3], [], [4, 5]]",
        schema={
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
    )
    assert result.to_list() == [[1, 2, 3], [], [4, 5]]
    assert str(result.type) == "3 * var * int64"

    result = ak.operations.from_json(
        " [ [ 1 ,2, 3], [], [4, 5]]" * 2,
        schema={
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        line_delimited=True,
    )
    assert result.to_list() == [[1, 2, 3], [], [4, 5]] * 2
    assert str(result.type) == "6 * var * int64"

    result = ak.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * var * int64"


def test_regulararray_integer():
    result = ak.operations.from_json(
        "[[1, 2, 3], [4, 5, 6]]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
    )
    assert result.to_list() == [[1, 2, 3], [4, 5, 6]]
    assert str(result.type) == "2 * 3 * int64"

    result = ak.operations.from_json(
        "[[1, 2, 3], [4, 5, 6]]" * 2,
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        line_delimited=True,
    )
    assert result.to_list() == [[1, 2, 3], [4, 5, 6]] * 2
    assert str(result.type) == "4 * 3 * int64"

    result = ak.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * 3 * int64"


def test_option_regulararray_integer():
    result = ak.operations.from_json(
        "[[1, 2, 3], null, [4, 5, 6]]",
        schema={
            "type": "array",
            "items": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
    )
    assert result.to_list() == [[1, 2, 3], None, [4, 5, 6]]
    assert str(result.type) == "3 * option[3 * int64]"

    result = ak.operations.from_json(
        "[[1, 2, 3], null, [4, 5, 6]]" * 2,
        schema={
            "type": "array",
            "items": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        line_delimited=True,
    )
    assert result.to_list() == [[1, 2, 3], None, [4, 5, 6]] * 2
    assert str(result.type) == "6 * option[3 * int64]"

    result = ak.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * option[3 * int64]"


def test_option_array_integer():
    result = ak.operations.from_json(
        " [ [ 1 ,2,3 ],null,[ ], [4, 5]]",
        schema={
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
    )
    assert result.to_list() == [[1, 2, 3], None, [], [4, 5]]
    assert str(result.type) == "4 * option[var * int64]"

    result = ak.operations.from_json(
        " [ [ 1 ,2,3 ],null,[ ], [4, 5]]" * 2,
        schema={
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
        line_delimited=True,
    )
    assert result.to_list() == [[1, 2, 3], None, [], [4, 5]] * 2
    assert str(result.type) == "8 * option[var * int64]"

    result = ak.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * option[var * int64]"


def test_option_array_option_integer():
    result = ak.operations.from_json(
        " [ [ 1 ,2,3 ],null,[ ] ,[null, 5]]",
        schema={
            "type": "array",
            "items": {
                "type": ["null", "array"],
                "items": {"type": ["integer", "null"]},
            },
        },
    )
    assert result.to_list() == [[1, 2, 3], None, [], [None, 5]]
    assert str(result.type) == "4 * option[var * ?int64]"

    result = ak.operations.from_json(
        " [ [ 1 ,2,3 ],null,[ ] ,[null, 5]]" * 2,
        schema={
            "type": "array",
            "items": {
                "type": ["null", "array"],
                "items": {"type": ["integer", "null"]},
            },
        },
        line_delimited=True,
    )
    assert result.to_list() == [[1, 2, 3], None, [], [None, 5]] * 2
    assert str(result.type) == "8 * option[var * ?int64]"

    result = ak.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {
                "type": ["null", "array"],
                "items": {"type": ["integer", "null"]},
            },
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * option[var * ?int64]"


def test_array_array_integer():
    result = ak.operations.from_json(
        " [ [ [ 1 ,2,3 ] ] ,[ [], [4, 5]], []]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
    )
    assert result.to_list() == [[[1, 2, 3]], [[], [4, 5]], []]
    assert str(result.type) == "3 * var * var * int64"

    result = ak.operations.from_json(
        " [ [ [ 1 ,2,3 ] ] ,[ [], [4, 5]], []]" * 2,
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
        line_delimited=True,
    )
    assert result.to_list() == [[[1, 2, 3]], [[], [4, 5]], []] * 2
    assert str(result.type) == "6 * var * var * int64"

    result = ak.operations.from_json(
        " [ [  ] ,[ ], []]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
    )
    assert result.to_list() == [[], [], []]
    assert str(result.type) == "3 * var * var * int64"

    result = ak.operations.from_json(
        " [ ]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * var * var * int64"


def test_record():
    result = ak.operations.from_json(
        ' [ { "x" :1 ,"y":1.1},{"y": 2.2, "x": 2}, {"x": 3, "y": 3.3}]',
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.to_list() == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]
    assert str(result.type) == "3 * {x: int64, y: float64}"

    result = ak.operations.from_json(
        ' [ { "x" :1 ,"y":1.1},{"y": 2.2, "x": 2}, {"x": 3, "y": 3.3}]' * 2,
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
        line_delimited=True,
    )
    assert (
        result.to_list()
        == [
            {"x": 1, "y": 1.1},
            {"x": 2, "y": 2.2},
            {"x": 3, "y": 3.3},
        ]
        * 2
    )
    assert str(result.type) == "6 * {x: int64, y: float64}"

    result = ak.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * {x: int64, y: float64}"


def test_option_record():
    result = ak.operations.from_json(
        ' [ { "x" : 1 ,"y":1.1},null ,{"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]',
        schema={
            "type": "array",
            "items": {
                "type": ["object", "null"],
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.to_list() == [
        {"x": 1, "y": 1.1},
        None,
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]
    assert str(result.type) == "4 * ?{x: int64, y: float64}"

    result = ak.operations.from_json(
        ' [ { "x" : 1 ,"y":1.1},null ,{"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]' * 2,
        schema={
            "type": "array",
            "items": {
                "type": ["object", "null"],
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
        line_delimited=True,
    )
    assert (
        result.to_list()
        == [
            {"x": 1, "y": 1.1},
            None,
            {"x": 2, "y": 2.2},
            {"x": 3, "y": 3.3},
        ]
        * 2
    )
    assert str(result.type) == "8 * ?{x: int64, y: float64}"

    result = ak.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {
                "type": ["object", "null"],
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.to_list() == []
    assert str(result.type) == "0 * ?{x: int64, y: float64}"


def test_top_record():
    result = ak.operations.from_json(
        ' { "x" :1 ,"y":1.1}  ',
        schema={
            "type": "object",
            "properties": {"y": {"type": "number"}, "x": {"type": "integer"}},
            "required": ["x", "y"],
        },
    )
    assert result.to_list() == {"x": 1, "y": 1.1}

    result = ak.operations.from_json(
        ' { "x" :1 ,"y":1.1}  ' * 2,
        schema={
            "type": "object",
            "properties": {"y": {"type": "number"}, "x": {"type": "integer"}},
            "required": ["x", "y"],
        },
        line_delimited=True,
    )
    assert result.to_list() == [{"x": 1, "y": 1.1}] * 2

    result = ak.operations.from_json(
        "",
        schema={
            "type": "object",
            "properties": {"y": {"type": "number"}, "x": {"type": "integer"}},
            "required": ["x", "y"],
        },
        line_delimited=True,
    )
    assert result.to_list() == []


def test_number_substitutions():
    result = ak.operations.from_json(
        '[1, 2, 3.14, "nan", "-inf", "inf", 999]',
        schema={"type": "array", "items": {"type": "number"}},
        nan_string="nan",
        posinf_string="inf",
        neginf_string="-inf",
    )
    assert result.to_list()[:3] == [1, 2, 3.14]
    assert np.isnan(result[3])
    assert result.to_list()[4:] == [-np.inf, np.inf, 999]
    assert str(result.type) == "7 * float64"

    result = ak.operations.from_json(
        '["nan", "-inf", "inf"]',
        schema={"type": "array", "items": {"type": "number"}},
        nan_string="nan",
        posinf_string="inf",
        neginf_string="-inf",
    )
    assert np.isnan(result[0])
    assert result.to_list()[1:] == [-np.inf, np.inf]
    assert str(result.type) == "3 * float64"

    result = ak.operations.from_json(
        '["nan", "-inf", "inf"]',
        schema={"type": "array", "items": {"type": "string"}},
        nan_string="nan",
        posinf_string="inf",
        neginf_string="-inf",
    )
    assert result.to_list() == ["nan", "-inf", "inf"]
    assert str(result.type) == "3 * string"


def test_complex_substitutions():
    result = ak.operations.from_json(
        '[{"r": 1, "i": 1.1}, {"r": 2, "i": 2.2}, {"r": "inf", "i": 3.3}]',
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"r": {"type": "number"}, "i": {"type": "number"}},
                "required": ["r", "i"],
            },
        },
        posinf_string="inf",
        complex_record_fields=("r", "i"),
    )
    assert result.to_list() == [1 + 1.1j, 2 + 2.2j, np.inf + 3.3j]
    assert str(result.type) == "3 * complex128"

    result = ak.operations.from_json(
        '[{"r": 1, "i": 1.1, "other": 1}, {"r": 2, "i": 2.2, "other": 1}, {"r": "inf", "i": 3.3, "other": 1}]',
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "r": {"type": "number"},
                    "i": {"type": "number"},
                    "other": {"type": "integer"},
                },
                "required": ["r", "i", "other"],
            },
        },
        posinf_string="inf",
        complex_record_fields=("r", "i"),
    )
    assert result.to_list() == [1 + 1.1j, 2 + 2.2j, np.inf + 3.3j]
    assert str(result.type) == "3 * complex128"


def test_ignore_before():
    for what in [
        "null",
        "true",
        "2",
        "2.2",
        "[]",
        "[2]",
        "[2, 2.2]",
        "{}",
        '{"z": 2.2}',
        '{"z": []}',
        '{"z": [2]}',
        '{"z": [2, 2.2]}',
    ]:
        array = ak.from_json(
            '[{"y": ' + what + ', "x": 1}, {"x": 3}]',
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        )
        assert array.to_list() == [{"x": 1}, {"x": 3}]
        assert str(array.type) == "2 * {x: int64}"


def test_ignore_after():
    for what in [
        "null",
        "true",
        "2",
        "2.2",
        "[]",
        "[2]",
        "[2, 2.2]",
        "{}",
        '{"z": 2.2}',
        '{"z": []}',
        '{"z": [2]}',
        '{"z": [2, 2.2]}',
    ]:
        array = ak.from_json(
            '[{"x": 1, "y": ' + what + '}, {"x": 3}]',
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        )
        assert array.to_list() == [{"x": 1}, {"x": 3}]
        assert str(array.type) == "2 * {x: int64}"


def test_ignore_between():
    for what in [
        "null",
        "true",
        "2",
        "2.2",
        "[]",
        "[2]",
        "[2, 2.2]",
        "{}",
        '{"z": 2.2}',
        '{"z": []}',
        '{"z": [2]}',
        '{"z": [2, 2.2]}',
    ]:
        array = ak.from_json(
            '[{"x": 1, "y": ' + what + ', "z": true}, {"x": 3, "z": false}]',
            schema={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"z": {"type": "boolean"}, "x": {"type": "integer"}},
                    "required": ["z", "x"],
                },
            },
        )
        assert array.to_list() == [{"x": 1, "z": True}, {"x": 3, "z": False}]
        assert str(array.type) == "2 * {z: bool, x: int64}"


def test_option_ignore_before():
    for what in [
        "null",
        "true",
        "2",
        "2.2",
        "[]",
        "[2]",
        "[2, 2.2]",
        "{}",
        '{"z": 2.2}',
        '{"z": []}',
        '{"z": [2]}',
        '{"z": [2, 2.2]}',
    ]:
        array = ak.from_json(
            '[{"y": ' + what + ', "x": 1}, {"x": 3}]',
            schema={
                "type": "array",
                "items": {
                    "type": ["object", "null"],
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        )
        assert array.to_list() == [{"x": 1}, {"x": 3}]
        assert str(array.type) == "2 * ?{x: int64}"


def test_option_ignore_after():
    for what in [
        "null",
        "true",
        "2",
        "2.2",
        "[]",
        "[2]",
        "[2, 2.2]",
        "{}",
        '{"z": 2.2}',
        '{"z": []}',
        '{"z": [2]}',
        '{"z": [2, 2.2]}',
    ]:
        array = ak.from_json(
            '[{"x": 1, "y": ' + what + '}, {"x": 3}]',
            schema={
                "type": "array",
                "items": {
                    "type": ["object", "null"],
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        )
        assert array.to_list() == [{"x": 1}, {"x": 3}]
        assert str(array.type) == "2 * ?{x: int64}"


def test_option_ignore_between():
    for what in [
        "null",
        "true",
        "2",
        "2.2",
        "[]",
        "[2]",
        "[2, 2.2]",
        "{}",
        '{"z": 2.2}',
        '{"z": []}',
        '{"z": [2]}',
        '{"z": [2, 2.2]}',
    ]:
        array = ak.from_json(
            '[{"x": 1, "y": ' + what + ', "z": true}, {"x": 3, "z": false}]',
            schema={
                "type": "array",
                "items": {
                    "type": ["object", "null"],
                    "properties": {"z": {"type": "boolean"}, "x": {"type": "integer"}},
                    "required": ["z", "x"],
                },
            },
        )
        assert array.to_list() == [{"x": 1, "z": True}, {"x": 3, "z": False}]
        assert str(array.type) == "2 * ?{z: bool, x: int64}"


def test_duplicate_keys():
    result = ak.operations.from_json(
        ' [ { "x" :1 ,"y":1.1, "x": 999},{"y": 2.2, "y": 999, "x": 2}, {"x": 3, "x": 999, "y": 3.3}]',
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
    )
    assert result.to_list() == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]
    assert str(result.type) == "3 * {x: int64, y: float64}"


def test_missing_optional_fields():
    result = ak.operations.from_json(
        ' [ { "y":1.1},{"y": 2.2, "x": 2}, {"x": 3}]',
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": ["integer", "null"]},
                    "y": {"type": ["number", "null"]},
                },
                "required": ["x", "y"],
            },
        },
    )
    assert result.to_list() == [
        {"x": None, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": None},
    ]
    assert str(result.type) == "3 * {x: ?int64, y: ?float64}"


def test_100_fields():
    # The bitwise checklist is defined in terms of 64-bit chunks (uint64_t),
    # and 100 > 64.

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                **{f"x{i:02d}": {"type": "integer"} for i in range(50)},
                **{f"x{i:02d}": {"type": ["integer", "null"]} for i in range(50, 100)},
            },
            "required": ["x", "y"],
        },
    }

    indata = [
        {f"x{i:02d}": 1 for i in range(100)},
        {f"x{i:02d}": 2 for i in range(100)},
        {f"x{i:02d}": 3 for i in range(100)},
    ]
    outdata = [
        {f"x{i:02d}": 1 for i in range(100)},
        {f"x{i:02d}": 2 for i in range(100)},
        {f"x{i:02d}": 3 for i in range(100)},
    ]
    assert ak.from_json(json.dumps(indata), schema=schema).to_list() == outdata

    del indata[1]["x57"]
    outdata[1]["x57"] = None
    assert ak.from_json(json.dumps(indata), schema=schema).to_list() == outdata

    del indata[1]["x99"]
    outdata[1]["x99"] = None
    assert ak.from_json(json.dumps(indata), schema=schema).to_list() == outdata

    del indata[0]["x75"]
    outdata[0]["x75"] = None
    assert ak.from_json(json.dumps(indata), schema=schema).to_list() == outdata

    del indata[2]["x57"]
    outdata[2]["x57"] = None
    assert ak.from_json(json.dumps(indata), schema=schema).to_list() == outdata

    del indata[1]["x25"]
    outdata[1]["x25"] = None
    with pytest.raises(ValueError) as err:
        ak.from_json(json.dumps(indata), schema=schema)
    assert "JSON schema mismatch" in str(err)

    del indata[1]["x49"]
    outdata[1]["x49"] = None
    with pytest.raises(ValueError) as err:
        ak.from_json(json.dumps(indata), schema=schema)
    assert "JSON schema mismatch" in str(err)

    del indata[2]["x00"]
    outdata[2]["x00"] = None
    with pytest.raises(ValueError) as err:
        ak.from_json(json.dumps(indata), schema=schema)
    assert "JSON schema mismatch" in str(err)


def test_complex_nested():
    # Multiple records have to manage different missing field checklists, so this tests that.
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {
                    "pull_request": {
                        "type": ["object", "null"],
                        "properties": {"merged_at": {"type": ["string", "null"]}},
                    }
                },
            }
        },
    }

    with open(os.path.join(SAMPLES_DIR, "complex-nested.json"), "rb") as file:
        array = ak.from_json(file, line_delimited=True, schema=schema)

    assert array["payload", "pull_request", "merged_at"][:5].to_list() == [
        None,
        None,
        None,
        None,
        None,
    ]

    assert array["payload", "pull_request", "merged_at"][
        [77, 169, 170, 172, 186, 207, 208]
    ].to_list() == [
        "2015-01-01T10:00:32Z",
        "2015-01-01T10:01:07Z",
        "2015-01-01T10:01:08Z",
        "2015-01-01T10:01:08Z",
        "2015-01-01T10:01:11Z",
        "2015-01-01T10:01:23Z",
        "2015-01-01T10:01:25Z",
    ]
