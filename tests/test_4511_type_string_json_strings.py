# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak
from awkward.types import NumpyType, RecordType
from awkward.types._awkward_datashape_parser import UnexpectedInput

# Each template puts a quoted string where the grammar reads a JSON string:
# a record key, a named-record key, a parameter key, and a parameter value.
TEMPLATES = [
    "{{{}: int64}}",
    "R[{}: int64]",
    "int8[parameters={{{}: 1}}]",
    'int8[parameters={{"k": {}}}]',
    'int8[parameters={{"k": [{}]}}]',
]

# Strings that are not valid JSON strings.
INVALID = [
    '"\x1f"',  # control character
    '"a\x00b"',  # control character
    '"a\tb"',  # unescaped tab
    r'"a\qb"',  # unknown escape
    r'"a\x41"',  # unknown escape
    r'"a\u12"',  # truncated \u escape
    r'"a\u12g4"',  # non-hex digit in \u escape
]


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("string", INVALID)
def test_invalid_json_string_is_a_parser_error(template, string):
    with pytest.raises(UnexpectedInput):
        ak.types.from_datashape(template.format(string), highlevel=False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            r'{"a\"b\\c\/d": int64}',
            RecordType([NumpyType("int64")], ['a"b\\c/d']),
        ),
        (
            r'R["\b\f\n\r\t": int64]',
            RecordType(
                [NumpyType("int64")], ["\b\f\n\r\t"], parameters={"__record__": "R"}
            ),
        ),
        (
            r'int8[parameters={"\u00e9\u00E9": "\u03bc"}]',
            NumpyType("int8", parameters={"\u00e9\u00e9": "\u03bc"}),
        ),
        (
            'int8[parameters={"\u00e9\x7f": ["\U0001f600"]}]',
            NumpyType("int8", parameters={"\u00e9\x7f": ["\U0001f600"]}),
        ),
        (
            'int8[parameters={"": ""}]',
            NumpyType("int8", parameters={"": ""}),
        ),
    ],
)
def test_valid_json_string_is_read(text, expected):
    assert ak.types.from_datashape(text, highlevel=False) == expected
