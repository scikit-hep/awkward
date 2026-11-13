# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward.types import ListType, NumpyType, OptionType, RegularType, UnknownType

STRING = ListType(
    NumpyType("uint8", parameters={"__array__": "char"}),
    parameters={"__array__": "string"},
)

CASES = [
    # the option is categorical
    (
        OptionType(NumpyType("int64"), parameters={"__categorical__": True}),
        "categorical[type=?int64]",
    ),
    (
        OptionType(ListType(NumpyType("int64")), parameters={"__categorical__": True}),
        "categorical[type=option[var * int64]]",
    ),
    (
        OptionType(
            RegularType(UnknownType(), 10), parameters={"__categorical__": True}
        ),
        "categorical[type=option[10 * unknown]]",
    ),
    (
        OptionType(NumpyType("int64"), parameters={"__categorical__": True, "x": 1}),
        'categorical[type=option[int64, parameters={"x": 1}]]',
    ),
    (
        OptionType(UnknownType(), parameters={"__categorical__": True}),
        "categorical[type=?unknown]",
    ),
    (
        OptionType(STRING, parameters={"__categorical__": True}),
        "categorical[type=?string]",
    ),
    # the content is categorical
    (
        OptionType(NumpyType("int64", parameters={"__categorical__": True})),
        "?categorical[type=int64]",
    ),
    (
        OptionType(ListType(NumpyType("int64"), parameters={"__categorical__": True})),
        "option[categorical[type=var * int64]]",
    ),
    (
        OptionType(
            NumpyType("int64", parameters={"__categorical__": True}),
            parameters={"x": 1},
        ),
        'option[categorical[type=int64], parameters={"x": 1}]',
    ),
    # both are categorical
    (
        OptionType(
            NumpyType("int64", parameters={"__categorical__": True}),
            parameters={"__categorical__": True},
        ),
        "categorical[type=?categorical[type=int64]]",
    ),
]


@pytest.mark.parametrize(("type_", "text"), CASES)
def test_str(type_, text):
    assert str(type_) == text


@pytest.mark.parametrize(("type_", "text"), CASES)
def test_from_datashape(type_, text):
    assert ak.types.from_datashape(text, highlevel=False).is_equal_to(
        type_, all_parameters=True
    )


def test_nullable_categorical_array():
    layout = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 0])),
        ak.contents.NumpyArray(np.array([7])),
        parameters={"__array__": "categorical"},
    )
    assert str(ak.Array(layout).type) == "3 * categorical[type=?int64]"
