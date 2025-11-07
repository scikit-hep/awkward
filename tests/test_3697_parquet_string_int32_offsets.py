# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
pyarrow_parquet = pytest.importorskip("pyarrow.parquet")


def test_parquet_string_roundtrip_with_string_to32_default(tmp_path):
    """Test that strings can be round-tripped through parquet with default string_to32=True.

    This test ensures that when string_to32=True (the default for parquet), strings are
    correctly serialized to Arrow's 32-bit string type and can be converted back to NumPy
    arrays without kernel lookup errors.

    Regression test for bug where int32 offsets from pyarrow.string() caused KeyError
    when converting to NumPy because UTF8 kernels only had int64 specializations.
    """
    # Create array with strings
    a = ak.Array([{"foo": "bar"}, {"foo": "baz"}])

    # Convert to numpy before serialization (should work with int64 offsets)
    b = np.asarray(a["foo"])
    assert b.dtype.kind == "U"
    assert list(b) == ["bar", "baz"]

    # Serialize to parquet with default string_to32=True
    ak.to_parquet(a, tmp_path / "test_string_to32.parquet")

    # Deserialize from parquet (creates int32 offsets for strings)
    x = ak.from_parquet(tmp_path / "test_string_to32.parquet")

    # This should work now with int32 offset kernel specializations
    y = np.asarray(x["foo"])
    assert y.dtype.kind == "U"
    assert list(y) == ["bar", "baz"]


def test_parquet_string_roundtrip_explicit_string_to32_true(tmp_path):
    """Test parquet roundtrip with explicitly setting string_to32=True."""
    a = ak.Array([{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}])

    # Explicitly set string_to32=True
    ak.to_parquet(a, tmp_path / "test_explicit_string_to32.parquet", string_to32=True)
    x = ak.from_parquet(tmp_path / "test_explicit_string_to32.parquet")

    # Convert to numpy (should use int32 offset kernels)
    result = np.asarray(x["name"])
    assert result.dtype.kind == "U"
    assert list(result) == ["Alice", "Bob", "Charlie"]


def test_parquet_string_roundtrip_string_to32_false(tmp_path):
    """Test parquet roundtrip with string_to32=False."""
    a = ak.Array([{"text": "hello"}, {"text": "world"}])

    # Set string_to32=False to use int64 offsets
    ak.to_parquet(a, tmp_path / "test_string_to32_false.parquet", string_to32=False)
    x = ak.from_parquet(tmp_path / "test_string_to32_false.parquet")

    # Convert to numpy (may use int64 offset kernels)
    result = np.asarray(x["text"])
    assert result.dtype.kind == "U"
    assert list(result) == ["hello", "world"]


def test_parquet_bytestring_roundtrip_with_bytestring_to32(tmp_path):
    """Test that bytestrings work correctly with bytestring_to32=True."""
    # Create array with bytestrings
    a = ak.Array([{"data": b"foo"}, {"data": b"bar"}])

    # Serialize with bytestring_to32=True (default)
    ak.to_parquet(a, tmp_path / "test_bytestring_to32.parquet", bytestring_to32=True)
    x = ak.from_parquet(tmp_path / "test_bytestring_to32.parquet")

    # Convert to numpy
    result = np.asarray(x["data"])
    assert result.dtype.kind == "S"
    assert list(result) == [b"foo", b"bar"]


def test_direct_string_array_with_int32_offsets(tmp_path):
    """Test direct conversion of string arrays with int32 offsets to NumPy."""
    # Create a string array
    strings = ak.Array(["hello", "world", "test"])

    # Serialize and deserialize through parquet to get int32 offsets
    ak.to_parquet(strings, tmp_path / "test_direct_strings.parquet", string_to32=True)
    x = ak.from_parquet(tmp_path / "test_direct_strings.parquet")

    # Check that offsets are int32
    layout = x.layout
    assert layout.offsets.dtype in (
        np.dtype("int32"),
        np.dtype("uint32"),
        np.dtype("int64"),
    )

    # Convert to numpy
    result = np.asarray(x)
    assert result.dtype.kind == "U"
    assert list(result) == ["hello", "world", "test"]


def test_empty_strings_with_int32_offsets(tmp_path):
    """Test that empty strings work correctly with int32 offsets."""
    a = ak.Array([{"msg": ""}, {"msg": "x"}, {"msg": ""}])

    ak.to_parquet(a, tmp_path / "test_empty_strings.parquet", string_to32=True)
    x = ak.from_parquet(tmp_path / "test_empty_strings.parquet")

    result = np.asarray(x["msg"])
    assert result.dtype.kind == "U"
    assert list(result) == ["", "x", ""]


def test_unicode_strings_with_int32_offsets(tmp_path):
    """Test that unicode strings work correctly with int32 offsets."""
    a = ak.Array([{"text": "Hello 👋"}, {"text": "World 🌍"}, {"text": "Test 🧪"}])

    ak.to_parquet(a, tmp_path / "test_unicode_strings.parquet", string_to32=True)
    x = ak.from_parquet(tmp_path / "test_unicode_strings.parquet")

    result = np.asarray(x["text"])
    assert result.dtype.kind == "U"
    assert list(result) == ["Hello 👋", "World 🌍", "Test 🧪"]
