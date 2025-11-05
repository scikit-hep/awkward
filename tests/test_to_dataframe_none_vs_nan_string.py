# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pandas = pytest.importorskip("pandas")


def test_none_vs_nan_string():
    """Test that None and 'nan' string are distinguishable in to_dataframe."""
    # Create array with None and literal "nan" string
    array = ak.Array([None, "nan"])
    df = ak.to_dataframe(array)
    result = df.to_dict()

    # None should be None, not the string "nan"
    assert result["values"][0] is None
    assert result["values"][1] == "nan"
    # They should be distinguishable
    assert result["values"][0] != result["values"][1]


def test_none_vs_nan_bytestring():
    """Test that None and b'nan' bytestring are distinguishable in to_dataframe."""
    # Create array with None and literal b"nan" bytestring
    array = ak.Array([None, b"nan"])
    df = ak.to_dataframe(array)
    result = df.to_dict()

    # None should be None, not the bytestring b"nan"
    assert result["values"][0] is None
    assert result["values"][1] == b"nan"
    # They should be distinguishable
    assert result["values"][0] != result["values"][1]


def test_nested_list_with_none_and_nan_string():
    """Test nested lists containing both None and 'nan' string."""
    array = ak.Array([["a", None, "nan"], ["b", "nan", None]])
    df = ak.to_dataframe(array)

    # Check that None and "nan" are distinguishable
    values = df["values"].values
    assert values[0] == "a"
    assert values[1] is None
    assert values[2] == "nan"
    assert values[3] == "b"
    assert values[4] == "nan"
    assert values[5] is None


def test_record_with_none_and_nan_string():
    """Test records containing both None and 'nan' string."""
    array = ak.Array([
        {"x": "value", "y": None},
        {"x": "nan", "y": "another"},
        {"x": None, "y": "nan"}
    ])
    df = ak.to_dataframe(array)

    # Check x column
    assert df["x"].values[0] == "value"
    assert df["x"].values[1] == "nan"
    assert df["x"].values[2] is None

    # Check y column
    assert df["y"].values[0] is None
    assert df["y"].values[1] == "another"
    assert df["y"].values[2] == "nan"
