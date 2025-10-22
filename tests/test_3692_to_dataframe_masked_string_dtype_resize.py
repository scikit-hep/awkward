# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pandas = pytest.importorskip("pandas")


def test_masked_string_array_with_option():
    """Test that string arrays with None values convert properly to dataframe."""
    # Create a string array with None values
    # This will create masked arrays internally during conversion
    array_str = ak.Array([["a", None, "c"], ["d", "e", None]])

    # Convert to dataframe - should not raise an error about dtype width
    df = ak.operations.to_dataframe(array_str)

    # Verify the conversion worked and "nan" appears for None values
    assert df["values"].values[0] == "a"
    assert df["values"].values[1] == "nan"
    assert df["values"].values[2] == "c"
    assert df["values"].values[3] == "d"
    assert df["values"].values[4] == "e"
    assert df["values"].values[5] == "nan"


def test_masked_bytestring_array_with_option():
    """Test that bytestring arrays with None values convert properly to dataframe."""
    # Create a bytestring array with None values
    array_bytes = ak.Array([[b"x", None, b"z"], [b"a", b"b", None]])

    # Convert to dataframe - should not raise an error about dtype width
    df = ak.operations.to_dataframe(array_bytes)

    # Verify the conversion worked and b"nan" appears for None values
    assert df["values"].values[0] == b"x"
    assert df["values"].values[1] == b"nan"
    assert df["values"].values[2] == b"z"
    assert df["values"].values[3] == b"a"
    assert df["values"].values[4] == b"b"
    assert df["values"].values[5] == b"nan"


def test_union_with_narrow_strings():
    """Test union arrays with narrow strings and None values."""
    # Test case from the existing test_union_to_record
    unionarray = ak.Array(
        [{"x": "a", "y": 1.1}, {"y": 2.2, "z": "b"}, {"x": "c", "y": 3.3}]
    )

    df = ak.operations.to_dataframe(unionarray)

    # Verify conversion works without dtype errors
    assert len(df) == 3
    # The x column should have "nan" for missing values
    assert df["x"].values[1] == "nan"


def test_single_char_strings_with_none():
    """Test very short strings with None values."""
    # Single character strings with None values
    # This is the edge case that needs dtype resizing
    array = ak.Array([["a", "b"], [None, "c"], ["d", None]])

    df = ak.operations.to_dataframe(array)

    # Verify all values are present
    values = df["values"].values.tolist()
    assert "a" in values
    assert "b" in values
    assert "c" in values
    assert "d" in values
    # Check that "nan" string appears for None values
    assert values.count("nan") == 2


def test_single_byte_bytestrings_with_none():
    """Test very short bytestrings with None values."""
    # Single byte strings with None values
    array = ak.Array([[b"a", b"b"], [None, b"c"], [b"d", None]])

    df = ak.operations.to_dataframe(array)

    # Verify all values are present
    values = df["values"].values.tolist()
    assert b"a" in values
    assert b"b" in values
    assert b"c" in values
    assert b"d" in values
    # Check that b"nan" appears for None values
    assert values.count(b"nan") == 2


def test_two_char_strings_with_none():
    """Test two-character strings with None values (edge case)."""
    # Two character strings - exactly the edge case where "nan" (3 chars) won't fit
    array = ak.Array([["ab", None], [None, "cd"]])

    df = ak.operations.to_dataframe(array)

    values = df["values"].values.tolist()
    assert "ab" in values
    assert "cd" in values
    # "nan" should appear (3 characters)
    assert values.count("nan") == 2
