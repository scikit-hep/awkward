# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak
from awkward.types import ListType, NumpyType, OptionType, UnknownType
from awkward.types._awkward_datashape_parser import UnexpectedInput


@pytest.mark.parametrize(
    "text",
    ["unknown", "unknown[parameters={}]", "[unknown, parameters={}]"],
)
def test_unknown_without_parameters(text):
    result = ak.types.from_datashape(text, highlevel=False)
    assert result == UnknownType()
    assert result.parameters == {}


@pytest.mark.parametrize(
    "text",
    [
        'unknown[parameters={"a": 1}]',
        'unknown[parameters={"a": null}]',
        '?unknown[parameters={"a": 1}]',
        'var * unknown[parameters={"a": 1}]',
    ],
)
def test_unknown_with_parameters_is_a_parser_error(text):
    with pytest.raises(UnexpectedInput):
        ak.types.from_datashape(text, highlevel=False)


@pytest.mark.parametrize(
    "text",
    [
        '[unknown, parameters={"a": 1}]',
        '?[unknown, parameters={"a": 1}]',
        'var * [unknown, parameters={"a": 1}]',
    ],
)
def test_list_parameters_on_unknown_raises(text):
    with pytest.raises(TypeError, match="UnknownType cannot contain parameters"):
        ak.types.from_datashape(text, highlevel=False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '[int64[parameters={"a": 1}], parameters={"b": 2}]',
            NumpyType("int64", parameters={"a": 1, "b": 2}),
        ),
        (
            '[int64[parameters={"a": 1}], parameters={"a": 2}]',
            NumpyType("int64", parameters={"a": 2}),
        ),
        (
            '[var * int64, parameters={"a": 1}]',
            ListType(NumpyType("int64"), parameters={"a": 1}),
        ),
        (
            '[?int64, parameters={"a": 1}]',
            OptionType(NumpyType("int64"), parameters={"a": 1}),
        ),
    ],
)
def test_list_parameters_on_other_types(text, expected):
    result = ak.types.from_datashape(text, highlevel=False)
    assert result == expected
    assert result.parameters == expected.parameters
