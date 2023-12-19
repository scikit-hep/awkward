# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pa = pytest.importorskip("pyarrow")


def test_strings():
    array = pa.chunked_array([["foo", "bar"], ["blah", "bleh"]])
    ak_array = ak.from_arrow(array)
    assert ak_array.type == ak.types.ArrayType(
        ak.types.ListType(
            ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
            parameters={"__array__": "string"},
        ),
        4,
    )


def test_strings_option():
    array = pa.chunked_array([["foo", "bar"], ["blah", "bleh", None]])
    ak_array = ak.from_arrow(array)
    assert ak_array.type == ak.types.ArrayType(
        ak.types.OptionType(
            ak.types.ListType(
                ak.types.NumpyType("uint8", parameters={"__array__": "char"}),
                parameters={"__array__": "string"},
            )
        ),
        5,
    )


def test_numbers():
    array = pa.chunked_array([[1, 2, 3], [4, 5]])
    ak_array = ak.from_arrow(array)
    assert ak_array.type == ak.types.ArrayType(ak.types.NumpyType("int64"), 5)


def test_numbers_option():
    array = pa.chunked_array([[1, 2, 3], [4, 5, None]])
    ak_array = ak.from_arrow(array)
    assert ak_array.type == ak.types.ArrayType(
        ak.types.OptionType(ak.types.NumpyType("int64")), 6
    )
