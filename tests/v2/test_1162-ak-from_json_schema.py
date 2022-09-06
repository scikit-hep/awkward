# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_boolean():
    result = ak._v2.operations.from_json(
        " [ true ,false, true, true, false]    ",
        schema={"type": "array", "items": {"type": "boolean"}},
    )
    assert result.tolist() == [True, False, True, True, False]
    assert str(result.type) == "5 * bool"

    result = ak._v2.operations.from_json(
        " [ true ,false, true, true, false]    " * 2,
        schema={"type": "array", "items": {"type": "boolean"}},
        line_delimited=True,
    )
    assert result.tolist() == [True, False, True, True, False] * 2
    assert str(result.type) == "10 * bool"

    result = ak._v2.operations.from_json(
        "",
        schema={"type": "array", "items": {"type": "boolean"}},
        line_delimited=True,
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * bool"

    result = ak._v2.operations.from_json(
        "[]",
        schema={"type": "array", "items": {"type": "boolean"}},
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * bool"


def test_integer():
    with pytest.raises(ValueError):
        ak._v2.operations.from_json(
            " [ 1 ,2 ,3.0, 4, 5]  \n  ",
            schema={"type": "array", "items": {"type": "integer"}},
        )

    result = ak._v2.operations.from_json(
        " [ 1 ,2 ,3, 4, 5]  \n  ",
        schema={"type": "array", "items": {"type": "integer"}},
    )
    assert result.tolist() == [1, 2, 3, 4, 5]
    assert str(result.type) == "5 * int64"

    result = ak._v2.operations.from_json(
        " [ 1 ,2 ,3, 4, 5]  \n  " * 2,
        schema={"type": "array", "items": {"type": "integer"}},
        line_delimited=True,
    )
    assert result.tolist() == [1, 2, 3, 4, 5] * 2
    assert str(result.type) == "10 * int64"

    result = ak._v2.operations.from_json(
        "[ ]",
        schema={"type": "array", "items": {"type": "integer"}},
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * int64"


def test_number():
    result = ak._v2.operations.from_json(
        " [ 1 ,2,3.14, 4, 5]",
        schema={"type": "array", "items": {"type": "number"}},
    )
    assert result.tolist() == [1, 2, 3.14, 4, 5]
    assert str(result.type) == "5 * float64"

    result = ak._v2.operations.from_json(
        " [ 1 ,2,3.14, 4, 5]" * 2,
        schema={"type": "array", "items": {"type": "number"}},
        line_delimited=True,
    )
    assert result.tolist() == [1, 2, 3.14, 4, 5] * 2
    assert str(result.type) == "10 * float64"

    result = ak._v2.operations.from_json(
        " [ ]",
        schema={"type": "array", "items": {"type": "number"}},
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * float64"


def test_option_boolean():
    result = ak._v2.operations.from_json(
        " [ true ,false ,null , true, false]",
        schema={"type": "array", "items": {"type": ["boolean", "null"]}},
    )
    assert result.tolist() == [True, False, None, True, False]
    assert str(result.type) == "5 * ?bool"

    result = ak._v2.operations.from_json(
        " [ true ,false ,null , true, false]" * 2,
        schema={"type": "array", "items": {"type": ["boolean", "null"]}},
        line_delimited=True,
    )
    assert result.tolist() == [True, False, None, True, False] * 2
    assert str(result.type) == "10 * ?bool"

    result = ak._v2.operations.from_json(
        " [ ]",
        schema={"type": "array", "items": {"type": ["boolean", "null"]}},
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * ?bool"


def test_option_integer():
    result = ak._v2.operations.from_json(
        " [ 1 ,2,null,4, 5]",
        schema={"type": "array", "items": {"type": ["null", "integer"]}},
    )
    assert result.tolist() == [1, 2, None, 4, 5]
    assert str(result.type) == "5 * ?int64"

    result = ak._v2.operations.from_json(
        " [ 1 ,2,null,4, 5]" * 2,
        schema={"type": "array", "items": {"type": ["null", "integer"]}},
        line_delimited=True,
    )
    assert result.tolist() == [1, 2, None, 4, 5] * 2
    assert str(result.type) == "10 * ?int64"

    result = ak._v2.operations.from_json(
        " [ ]",
        schema={"type": "array", "items": {"type": ["null", "integer"]}},
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * ?int64"


def test_string():
    result = ak._v2.operations.from_json(
        r' [ "" ,"two","three \u2192 3", "\"four\"", "fi\nve"]',
        schema={"type": "array", "items": {"type": "string"}},
    )
    assert result.tolist() == ["", "two", "three \u2192 3", '"four"', "fi\nve"]
    assert str(result.type) == "5 * string"

    result = ak._v2.operations.from_json(
        r' [ "" ,"two","three \u2192 3", "\"four\"", "fi\nve"]' * 2,
        schema={"type": "array", "items": {"type": "string"}},
        line_delimited=True,
    )
    assert result.tolist() == ["", "two", "three \u2192 3", '"four"', "fi\nve"] * 2
    assert str(result.type) == "10 * string"

    result = ak._v2.operations.from_json(
        r"[]",
        schema={"type": "array", "items": {"type": "string"}},
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * string"


def test_option_string():
    result = ak._v2.operations.from_json(
        r' [ "" ,null ,"three \u2192 3", "\"four\"", "fi\nve"]',
        schema={"type": "array", "items": {"type": ["null", "string"]}},
    )
    assert result.tolist() == ["", None, "three \u2192 3", '"four"', "fi\nve"]
    assert str(result.type) == "5 * ?string"

    result = ak._v2.operations.from_json(
        r' [ "" ,null ,"three \u2192 3", "\"four\"", "fi\nve"]' * 2,
        schema={"type": "array", "items": {"type": ["null", "string"]}},
        line_delimited=True,
    )
    assert result.tolist() == ["", None, "three \u2192 3", '"four"', "fi\nve"] * 2
    assert str(result.type) == "10 * ?string"

    result = ak._v2.operations.from_json(
        r"[]",
        schema={"type": "array", "items": {"type": ["null", "string"]}},
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * ?string"


def test_enum_string():
    result = ak._v2.operations.from_json(
        r'["three", "two", "one", "one", "two", "three"]',
        schema={
            "type": "array",
            "items": {"type": "string", "enum": ["one", "two", "three"]},
        },
    )
    assert result.tolist() == ["three", "two", "one", "one", "two", "three"]
    assert isinstance(result.layout, ak._v2.contents.IndexedArray)
    assert result.layout.index.data.tolist() == [2, 1, 0, 0, 1, 2]
    assert str(result.type) == "6 * categorical[type=string]"

    result = ak._v2.operations.from_json(
        r'["three", "two", "one", "one", "two", "three"]' * 2,
        schema={
            "type": "array",
            "items": {"type": "string", "enum": ["one", "two", "three"]},
        },
        line_delimited=True,
    )
    assert result.tolist() == ["three", "two", "one", "one", "two", "three"] * 2
    assert isinstance(result.layout, ak._v2.contents.IndexedArray)
    assert result.layout.index.data.tolist() == [2, 1, 0, 0, 1, 2] * 2
    assert str(result.type) == "12 * categorical[type=string]"

    result = ak._v2.operations.from_json(
        r"[]",
        schema={
            "type": "array",
            "items": {"type": "string", "enum": ["one", "two", "three"]},
        },
    )
    assert result.tolist() == []
    assert isinstance(result.layout, ak._v2.contents.IndexedArray)
    assert result.layout.index.data.tolist() == []
    assert str(result.type) == "0 * categorical[type=string]"


def test_option_enum_string():
    result = ak._v2.operations.from_json(
        r'["three", "two", null, "one", "one", "two", "three"]',
        schema={
            "type": "array",
            "items": {"type": ["null", "string"], "enum": ["one", "two", "three"]},
        },
    )
    assert result.tolist() == ["three", "two", None, "one", "one", "two", "three"]
    assert isinstance(result.layout, ak._v2.contents.IndexedOptionArray)
    assert result.layout.index.data.tolist() == [2, 1, -1, 0, 0, 1, 2]
    assert str(result.type) == "7 * ?categorical[type=string]"

    result = ak._v2.operations.from_json(
        r'["three", "two", null, "one", "one", "two", "three"]' * 2,
        schema={
            "type": "array",
            "items": {"type": ["null", "string"], "enum": ["one", "two", "three"]},
        },
        line_delimited=True,
    )
    assert result.tolist() == ["three", "two", None, "one", "one", "two", "three"] * 2
    assert isinstance(result.layout, ak._v2.contents.IndexedOptionArray)
    assert result.layout.index.data.tolist() == [2, 1, -1, 0, 0, 1, 2] * 2
    assert str(result.type) == "14 * ?categorical[type=string]"

    result = ak._v2.operations.from_json(
        r"[]",
        schema={
            "type": "array",
            "items": {"type": ["null", "string"], "enum": ["one", "two", "three"]},
        },
    )
    assert result.tolist() == []
    assert isinstance(result.layout, ak._v2.contents.IndexedOptionArray)
    assert result.layout.index.data.tolist() == []
    assert str(result.type) == "0 * ?categorical[type=string]"


def test_array_integer():
    result = ak._v2.operations.from_json(
        " [ [ 1 ,2, 3], [], [4, 5]]",
        schema={
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
    )
    assert result.tolist() == [[1, 2, 3], [], [4, 5]]
    assert str(result.type) == "3 * var * int64"

    result = ak._v2.operations.from_json(
        " [ [ 1 ,2, 3], [], [4, 5]]" * 2,
        schema={
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        line_delimited=True,
    )
    assert result.tolist() == [[1, 2, 3], [], [4, 5]] * 2
    assert str(result.type) == "6 * var * int64"

    result = ak._v2.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        },
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * var * int64"


def test_regulararray_integer():
    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert str(result.type) == "2 * 3 * int64"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [[1, 2, 3], [4, 5, 6]] * 2
    assert str(result.type) == "4 * 3 * int64"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == []
    assert str(result.type) == "0 * 3 * int64"


def test_option_regulararray_integer():
    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [[1, 2, 3], None, [4, 5, 6]]
    assert str(result.type) == "3 * option[3 * int64]"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [[1, 2, 3], None, [4, 5, 6]] * 2
    assert str(result.type) == "6 * option[3 * int64]"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == []
    assert str(result.type) == "0 * option[3 * int64]"


def test_option_array_integer():
    result = ak._v2.operations.from_json(
        " [ [ 1 ,2,3 ],null,[ ], [4, 5]]",
        schema={
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
    )
    assert result.tolist() == [[1, 2, 3], None, [], [4, 5]]
    assert str(result.type) == "4 * option[var * int64]"

    result = ak._v2.operations.from_json(
        " [ [ 1 ,2,3 ],null,[ ], [4, 5]]" * 2,
        schema={
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
        line_delimited=True,
    )
    assert result.tolist() == [[1, 2, 3], None, [], [4, 5]] * 2
    assert str(result.type) == "8 * option[var * int64]"

    result = ak._v2.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {"type": ["null", "array"], "items": {"type": "integer"}},
        },
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * option[var * int64]"


def test_option_array_option_integer():
    result = ak._v2.operations.from_json(
        " [ [ 1 ,2,3 ],null,[ ] ,[null, 5]]",
        schema={
            "type": "array",
            "items": {
                "type": ["null", "array"],
                "items": {"type": ["integer", "null"]},
            },
        },
    )
    assert result.tolist() == [[1, 2, 3], None, [], [None, 5]]
    assert str(result.type) == "4 * option[var * ?int64]"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [[1, 2, 3], None, [], [None, 5]] * 2
    assert str(result.type) == "8 * option[var * ?int64]"

    result = ak._v2.operations.from_json(
        "[]",
        schema={
            "type": "array",
            "items": {
                "type": ["null", "array"],
                "items": {"type": ["integer", "null"]},
            },
        },
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * option[var * ?int64]"


def test_array_array_integer():
    result = ak._v2.operations.from_json(
        " [ [ [ 1 ,2,3 ] ] ,[ [], [4, 5]], []]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
    )
    assert result.tolist() == [[[1, 2, 3]], [[], [4, 5]], []]
    assert str(result.type) == "3 * var * var * int64"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [[[1, 2, 3]], [[], [4, 5]], []] * 2
    assert str(result.type) == "6 * var * var * int64"

    result = ak._v2.operations.from_json(
        " [ [  ] ,[ ], []]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
    )
    assert result.tolist() == [[], [], []]
    assert str(result.type) == "3 * var * var * int64"

    result = ak._v2.operations.from_json(
        " [ ]",
        schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
    )
    assert result.tolist() == []
    assert str(result.type) == "0 * var * var * int64"


def test_record():
    result = ak._v2.operations.from_json(
        ' [ { "x" :1 ,"y":1.1},{"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]',
        schema={
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
    assert str(result.type) == "3 * {x: int64, y: float64}"

    result = ak._v2.operations.from_json(
        ' [ { "x" :1 ,"y":1.1},{"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]' * 2,
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
        result.tolist()
        == [
            {"x": 1, "y": 1.1},
            {"x": 2, "y": 2.2},
            {"x": 3, "y": 3.3},
        ]
        * 2
    )
    assert str(result.type) == "6 * {x: int64, y: float64}"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == []
    assert str(result.type) == "0 * {x: int64, y: float64}"


def test_option_record():
    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [
        {"x": 1, "y": 1.1},
        None,
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]
    assert str(result.type) == "4 * ?{x: int64, y: float64}"

    result = ak._v2.operations.from_json(
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
        result.tolist()
        == [
            {"x": 1, "y": 1.1},
            None,
            {"x": 2, "y": 2.2},
            {"x": 3, "y": 3.3},
        ]
        * 2
    )
    assert str(result.type) == "8 * ?{x: int64, y: float64}"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == []
    assert str(result.type) == "0 * ?{x: int64, y: float64}"


def test_top_record():
    result = ak._v2.operations.from_json(
        ' { "x" :1 ,"y":1.1}  ',
        schema={
            "type": "object",
            "properties": {"y": {"type": "number"}, "x": {"type": "integer"}},
            "required": ["x", "y"],
        },
    )
    assert result.tolist() == {"x": 1, "y": 1.1}

    result = ak._v2.operations.from_json(
        ' { "x" :1 ,"y":1.1}  ' * 2,
        schema={
            "type": "object",
            "properties": {"y": {"type": "number"}, "x": {"type": "integer"}},
            "required": ["x", "y"],
        },
        line_delimited=True,
    )
    assert result.tolist() == [{"x": 1, "y": 1.1}] * 2

    result = ak._v2.operations.from_json(
        "",
        schema={
            "type": "object",
            "properties": {"y": {"type": "number"}, "x": {"type": "integer"}},
            "required": ["x", "y"],
        },
        line_delimited=True,
    )
    assert result.tolist() == []


def test_number_substitutions():
    result = ak._v2.operations.from_json(
        '[1, 2, 3.14, "nan", "-inf", "inf", 999]',
        schema={"type": "array", "items": {"type": "number"}},
        nan_string="nan",
        posinf_string="inf",
        neginf_string="-inf",
    )
    assert result.tolist()[:3] == [1, 2, 3.14]
    assert np.isnan(result[3])
    assert result.tolist()[4:] == [-np.inf, np.inf, 999]
    assert str(result.type) == "7 * float64"

    result = ak._v2.operations.from_json(
        '["nan", "-inf", "inf"]',
        schema={"type": "array", "items": {"type": "number"}},
        nan_string="nan",
        posinf_string="inf",
        neginf_string="-inf",
    )
    assert np.isnan(result[0])
    assert result.tolist()[1:] == [-np.inf, np.inf]
    assert str(result.type) == "3 * float64"

    result = ak._v2.operations.from_json(
        '["nan", "-inf", "inf"]',
        schema={"type": "array", "items": {"type": "string"}},
        nan_string="nan",
        posinf_string="inf",
        neginf_string="-inf",
    )
    assert result.tolist() == ["nan", "-inf", "inf"]
    assert str(result.type) == "3 * string"


def test_complex_substitutions():
    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [1 + 1.1j, 2 + 2.2j, np.inf + 3.3j]
    assert str(result.type) == "3 * complex128"

    result = ak._v2.operations.from_json(
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
    assert result.tolist() == [1 + 1.1j, 2 + 2.2j, np.inf + 3.3j]
    assert str(result.type) == "3 * complex128"
