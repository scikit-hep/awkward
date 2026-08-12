# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Extended tests for ak.to_cudf covering:
- All primitive dtypes (int8/16/32/64, uint8/16/32/64, float32/64, bool)
- Datetime and timedelta dtypes
- Empty arrays of various types
- Deeply nested structs (3+ levels)
- Structs with many fields
- Nullable struct fields (inner nulls)
- Nullable ragged (variable-length) lists
- Nullable list elements (list-of-optional)
- List-of-struct and struct-of-list combinations
- Tuple-style (unnamed/positional-field) records
- IndexedOptionArray path
- ListArray (explicit starts/stops, delegates to ListOffsetArray)
- ByteMaskedArray (explicit byte-mask construction)
- UnmaskedArray (passthrough to content)
- EmptyArray (zero-length fallback)
- IndexedArray (carry/project path)
- RegularArray -> NotImplementedError
- UnionArray -> NotImplementedError
- Round-trip: to_cudf -> from_cudf
"""

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cudf = pytest.importorskip("cudf", exc_type=ImportError)
cupy = pytest.importorskip("cupy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_arrow_list(series):
    """Convert a cudf Series to a plain Python list via Arrow."""
    return series.to_arrow().tolist()


# ---------------------------------------------------------------------------
# Primitive dtypes
# ---------------------------------------------------------------------------


def test_to_cudf_int8():
    arr = ak.Array(np.array([1, -1, 127, -128], dtype=np.int8))
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert _to_arrow_list(out) == [1, -1, 127, -128]
    assert out.dtype == np.dtype("int8")


def test_to_cudf_int16():
    arr = ak.Array(np.array([0, 32767, -32768, 100], dtype=np.int16))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [0, 32767, -32768, 100]
    assert out.dtype == np.dtype("int16")


def test_to_cudf_int32():
    arr = ak.Array(np.array([10, -10, 2**30], dtype=np.int32))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [10, -10, 2**30]
    assert out.dtype == np.dtype("int32")


def test_to_cudf_int64():
    arr = ak.Array(np.array([0, -(2**62), 2**62], dtype=np.int64))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [0, -(2**62), 2**62]
    assert out.dtype == np.dtype("int64")


def test_to_cudf_uint8():
    arr = ak.Array(np.array([0, 128, 255], dtype=np.uint8))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [0, 128, 255]
    assert out.dtype == np.dtype("uint8")


def test_to_cudf_uint16():
    arr = ak.Array(np.array([0, 1000, 65535], dtype=np.uint16))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [0, 1000, 65535]
    assert out.dtype == np.dtype("uint16")


def test_to_cudf_uint32():
    arr = ak.Array(np.array([0, 2**31, 2**32 - 1], dtype=np.uint32))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [0, 2**31, 2**32 - 1]
    assert out.dtype == np.dtype("uint32")


def test_to_cudf_uint64():
    arr = ak.Array(np.array([0, 2**63, 2**64 - 1], dtype=np.uint64))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [0, 2**63, 2**64 - 1]
    assert out.dtype == np.dtype("uint64")


def test_to_cudf_float32():
    arr = ak.Array(np.array([0.0, 1.5, -1.5, np.inf, -np.inf], dtype=np.float32))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[:3] == pytest.approx([0.0, 1.5, -1.5])
    assert result[3] == np.inf
    assert result[4] == -np.inf
    assert out.dtype == np.dtype("float32")


def test_to_cudf_float64():
    arr = ak.Array(np.array([0.0, 1.23456789, -9.87654321], dtype=np.float64))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == pytest.approx([0.0, 1.23456789, -9.87654321])
    assert out.dtype == np.dtype("float64")


def test_to_cudf_bool():
    arr = ak.Array(np.array([True, False, True, True, False], dtype=np.bool_))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [True, False, True, True, False]
    assert out.dtype == np.dtype("bool")


# ---------------------------------------------------------------------------
# Empty arrays
# ---------------------------------------------------------------------------


def test_to_cudf_empty_int64():
    arr = ak.Array(np.array([], dtype=np.int64))
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert _to_arrow_list(out) == []
    assert out.dtype == np.dtype("int64")


def test_to_cudf_empty_float64():
    arr = ak.Array(np.array([], dtype=np.float64))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == []


def test_to_cudf_empty_list():
    arr = ak.Array([[], [], []])
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert _to_arrow_list(out) == [[], [], []]


def test_to_cudf_empty_outer():
    # An array with zero entries of list type
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0], dtype=np.int64)),
            ak.contents.NumpyArray(np.array([], dtype=np.int32)),
        )
    )
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert _to_arrow_list(out) == []


# ---------------------------------------------------------------------------
# Nullable primitives — ByteMaskedArray and IndexedOptionArray
# ---------------------------------------------------------------------------


def test_to_cudf_nullable_all_valid():
    # IndexedOptionArray with no -1s: all entries valid, but option-typed
    index = ak.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([10, 20, 30, 40], dtype=np.int32))
    opt_arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(opt_arr)
    assert _to_arrow_list(out) == [10, 20, 30, 40]


def test_to_cudf_nullable_int8():
    # Build a nullable int8 array using IndexedOptionArray (-1 marks None)
    index = ak.index.Index64(np.array([0, -1, 2], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([1, 0, 3], dtype=np.int8))
    arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == 1
    assert result[1] is None
    assert result[2] == 3


def test_to_cudf_nullable_float32():
    # Build a nullable float32 array using IndexedOptionArray (-1 marks None)
    index = ak.index.Index64(np.array([-1, 0, -1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] is None
    assert result[1] == pytest.approx(2.0)
    assert result[2] is None


def test_to_cudf_nullable_bool():
    arr = ak.Array([True, None, False, None, True])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [True, None, False, None, True]


def test_to_cudf_nullable_leading_null():
    arr = ak.Array([None, 1, 2, 3])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [None, 1, 2, 3]


def test_to_cudf_nullable_trailing_null():
    arr = ak.Array([1, 2, 3, None])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [1, 2, 3, None]


def test_to_cudf_nullable_all_null():
    # All entries are null; use a typed IndexedOptionArray so cudf gets a
    # known dtype (int64) rather than an unresolvable EmptyForm.
    index = ak.index.Index64(np.array([-1, -1, -1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([], dtype=np.int64))
    arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [None, None, None]


# ---------------------------------------------------------------------------
# Ragged (variable-length) lists
# ---------------------------------------------------------------------------


def test_to_cudf_ragged_int32():
    arr = ak.Array([[1, 2, 3], [], [4], [5, 6]])
    # coerce content to int32
    arr = ak.values_astype(arr, np.int32)
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[1, 2, 3], [], [4], [5, 6]]


def test_to_cudf_ragged_float64():
    arr = ak.Array([[1.1, 2.2], [3.3], [], [4.4, 5.5, 6.6]])
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == pytest.approx([1.1, 2.2])
    assert result[1] == pytest.approx([3.3])
    assert result[2] == []
    assert result[3] == pytest.approx([4.4, 5.5, 6.6])


def test_to_cudf_ragged_bool():
    arr = ak.Array([[True, False], [True], [], [False, True, False]])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[True, False], [True], [], [False, True, False]]


def test_to_cudf_nullable_outer_list():
    # The outer list entries can be None
    arr = ak.Array([[1, 2], None, [3], None, []])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[1, 2], None, [3], None, []]


def test_to_cudf_list_of_nullable_ints():
    # Each list entry contains optional integers
    arr = ak.Array([[1, None, 3], [None, 5], []])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[1, None, 3], [None, 5], []]


def test_to_cudf_deeply_nested_ragged_3():
    # 3-level ragged: list-of-list-of-list-of-int
    arr = ak.Array([[[[1], [2, 3]], [[4]]], [[[5, 6, 7]]]])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[[[1], [2, 3]], [[4]]], [[[5, 6, 7]]]]


def test_to_cudf_deeply_nested_ragged_4():
    # 4-level ragged
    arr = ak.Array([[[[[1, 2], []], [[3]]]]])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[[[[1, 2], []], [[3]]]]]


# ---------------------------------------------------------------------------
# Struct layouts
# ---------------------------------------------------------------------------


def test_to_cudf_struct_all_dtypes():
    # Struct with fields of different dtypes
    arr = ak.Array(
        [
            {
                "i8": np.int8(1),
                "u32": np.uint32(2),
                "f32": np.float32(3.0),
                "flag": True,
            },
            {
                "i8": np.int8(-1),
                "u32": np.uint32(100),
                "f32": np.float32(-1.5),
                "flag": False,
            },
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0]["i8"] == 1
    assert result[0]["u32"] == 2
    assert result[0]["f32"] == pytest.approx(3.0)
    assert result[0]["flag"] is True
    assert result[1]["i8"] == -1
    assert result[1]["flag"] is False


def test_to_cudf_struct_many_fields():
    # Struct with 8 fields
    data = [
        {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8},
        {"a": 9, "b": 10, "c": 11, "d": 12, "e": 13, "f": 14, "g": 15, "h": 16},
    ]
    arr = ak.Array(data)
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == data


def test_to_cudf_struct_depth_3():
    # 3-level nested struct
    arr = ak.Array(
        [
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"c": 2}}},
            {"a": {"b": {"c": 3}}},
        ]
    )
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [
        {"a": {"b": {"c": 1}}},
        {"a": {"b": {"c": 2}}},
        {"a": {"b": {"c": 3}}},
    ]


def test_to_cudf_struct_depth_4():
    # 4-level nested struct
    arr = ak.Array(
        [
            {"w": {"x": {"y": {"z": 10}}}},
            {"w": {"x": {"y": {"z": 20}}}},
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0]["w"]["x"]["y"]["z"] == 10
    assert result[1]["w"]["x"]["y"]["z"] == 20


def test_to_cudf_struct_with_nullable_field():
    # Struct where one field has null values
    arr = ak.Array(
        [
            {"x": 1, "y": 1.0},
            {"x": 2, "y": None},
            {"x": 3, "y": 3.0},
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == {"x": 1, "y": 1.0}
    assert result[1]["x"] == 2
    assert result[1]["y"] is None
    assert result[2] == {"x": 3, "y": 3.0}


def test_to_cudf_struct_with_all_nullable_fields():
    arr = ak.Array(
        [
            {"a": None, "b": None},
            {"a": 1, "b": 2.0},
            {"a": None, "b": 3.0},
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0]["a"] is None
    assert result[0]["b"] is None
    assert result[1]["a"] == 1
    assert result[2]["a"] is None


def test_to_cudf_nullable_struct():
    # The struct record itself can be None
    arr = ak.Array([{"x": 1, "y": 2}, None, {"x": 3, "y": 4}])
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == {"x": 1, "y": 2}
    assert result[1] is None
    assert result[2] == {"x": 3, "y": 4}


def test_to_cudf_empty_struct():
    arr = ak.Array([{"x": 1}, {"x": 2}][:0])
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# List-of-struct and struct-of-list combinations
# ---------------------------------------------------------------------------


def test_to_cudf_list_of_struct():
    arr = ak.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
            [],
            [{"x": 3, "y": 3.3}],
        ]
    )
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        [],
        [{"x": 3, "y": 3.3}],
    ]


def test_to_cudf_struct_of_list():
    arr = ak.Array(
        [
            {"points": [1, 2, 3], "label": 0},
            {"points": [], "label": 1},
            {"points": [4, 5], "label": 2},
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0]["points"] == [1, 2, 3]
    assert result[0]["label"] == 0
    assert result[1]["points"] == []
    assert result[2]["points"] == [4, 5]


def test_to_cudf_struct_of_list_of_struct():
    arr = ak.Array(
        [
            {"hits": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
            {"hits": []},
            {"hits": [{"x": 5, "y": 6}]},
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0]["hits"] == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    assert result[1]["hits"] == []
    assert result[2]["hits"] == [{"x": 5, "y": 6}]


def test_to_cudf_list_of_struct_of_list():
    arr = ak.Array(
        [
            [{"vals": [1, 2]}, {"vals": [3]}],
            [{"vals": []}],
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0][0]["vals"] == [1, 2]
    assert result[0][1]["vals"] == [3]
    assert result[1][0]["vals"] == []


def test_to_cudf_nullable_list_of_struct():
    arr = ak.Array(
        [
            [{"a": 1}, {"a": 2}],
            None,
            [{"a": 3}],
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == [{"a": 1}, {"a": 2}]
    assert result[1] is None
    assert result[2] == [{"a": 3}]


# ---------------------------------------------------------------------------
# Strings (extended)
# ---------------------------------------------------------------------------


def test_to_cudf_empty_strings():
    arr = ak.Array(["", "a", "", "bb"])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == ["", "a", "", "bb"]


def test_to_cudf_string_list():
    arr = ak.Array([["hello", "world"], [], ["awkward"]])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [["hello", "world"], [], ["awkward"]]


def test_to_cudf_struct_with_string_field():
    arr = ak.Array(
        [
            {"name": "alice", "score": 10},
            {"name": "bob", "score": 20},
        ]
    )
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0]["name"] == "alice"
    assert result[1]["name"] == "bob"


# ---------------------------------------------------------------------------
# Round-trips: to_cudf -> from_cudf
# ---------------------------------------------------------------------------


def test_roundtrip_flat_int64():
    arr = ak.Array(np.array([10, 20, 30], dtype=np.int64))
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [10, 20, 30]
    assert ak.backend(result) == "cuda"


def test_roundtrip_ragged():
    arr = ak.Array([[1, 2, 3], [], [4, 5]])
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [[1, 2, 3], [], [4, 5]]


def test_roundtrip_nested_ragged():
    arr = ak.Array([[[1, 2], [3]], [[4]]])
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [[[1, 2], [3]], [[4]]]


def test_roundtrip_struct():
    arr = ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]


def test_roundtrip_nested_struct():
    arr = ak.Array([{"a": {"b": 1}}, {"a": {"b": 2}}])
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [{"a": {"b": 1}}, {"a": {"b": 2}}]


def test_roundtrip_nullable():
    # Roundtrip via a cudf Series: create nullable Series, convert to ak,
    # then back to cudf and verify values.  This avoids the to_cudf→from_cudf
    # path which is already covered by test_3948_from_cudf_extended.
    series = cudf.Series([1, None, 3, None, 5], dtype="Int64")
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [1, None, 3, None, 5]


def test_roundtrip_list_of_struct():
    arr = ak.Array(
        [
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            [],
            [{"x": 5, "y": 6}],
        ]
    )
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        [],
        [{"x": 5, "y": 6}],
    ]


def test_roundtrip_struct_of_list():
    arr = ak.Array(
        [
            {"vals": [1, 2, 3], "n": 3},
            {"vals": [], "n": 0},
            {"vals": [4], "n": 1},
        ]
    )
    result = ak.from_cudf(ak.to_cudf(arr))
    flat = ak.to_list(result)
    assert flat[0]["vals"] == [1, 2, 3]
    assert flat[1]["vals"] == []
    assert flat[2]["vals"] == [4]


def test_roundtrip_strings():
    arr = ak.Array(["hello", "world", "awkward"])
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == ["hello", "world", "awkward"]


def test_roundtrip_nullable_strings():
    series = cudf.Series(["hello", None, "awkward"])
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == ["hello", None, "awkward"]


def test_roundtrip_nullable_struct():
    series = cudf.Series([{"x": 1, "y": 2}, None, {"x": 3, "y": 4}])
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [{"x": 1, "y": 2}, None, {"x": 3, "y": 4}]


def test_roundtrip_nullable_list():
    series = cudf.Series([[1, 2], None, [3], None, []])
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [[1, 2], None, [3], None, []]


# ---------------------------------------------------------------------------
# BitMaskedArray paths (bitmask encoding)
# ---------------------------------------------------------------------------


def test_to_cudf_bitmasked_lsb_valid_when_true():
    nullable = ak.Array([1, None, 3, None, 5, None, 7, None])
    bit_arr = ak.Array(nullable.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(bit_arr)
    assert _to_arrow_list(out) == [1, None, 3, None, 5, None, 7, None]


def test_to_cudf_bitmasked_msb_valid_when_true():
    nullable = ak.Array([1, None, 3, None, 5, None, 7, None])
    bit_arr = ak.Array(nullable.layout.to_BitMaskedArray(True, False))
    out = ak.to_cudf(bit_arr)
    assert _to_arrow_list(out) == [1, None, 3, None, 5, None, 7, None]


def test_to_cudf_bitmasked_struct():
    # Struct with a nullable int field, bitmask-encoded
    nullable_field = ak.Array([10, None, 30, None])
    bit_field = ak.Array(nullable_field.layout.to_BitMaskedArray(True, True))
    struct_arr = ak.zip({"a": ak.Array([1, 2, 3, 4]), "b": bit_field})
    out = ak.to_cudf(struct_arr)
    result = _to_arrow_list(out)
    assert result[0]["b"] == 10
    assert result[1]["b"] is None
    assert result[2]["b"] == 30
    assert result[3]["b"] is None


# ---------------------------------------------------------------------------
# ListArray (explicit starts/stops) — delegates to ListOffsetArray64
# ---------------------------------------------------------------------------


def test_to_cudf_listarray_basic():
    # Build a ListArray explicitly with starts and stops buffers.
    # ListArray._to_cudf delegates to to_ListOffsetArray64(False)._to_cudf.
    starts = ak.index.Index64(np.array([0, 3, 3], dtype=np.int64))
    stops = ak.index.Index64(np.array([3, 3, 5], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    arr = ak.Array(ak.contents.ListArray(starts, stops, content))
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert _to_arrow_list(out) == [[1, 2, 3], [], [4, 5]]


def test_to_cudf_listarray_empty_lists():
    # All lists are empty.
    starts = ak.index.Index64(np.array([0, 0, 0], dtype=np.int64))
    stops = ak.index.Index64(np.array([0, 0, 0], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([], dtype=np.float64))
    arr = ak.Array(ak.contents.ListArray(starts, stops, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[], [], []]


def test_to_cudf_listarray_single_list():
    starts = ak.index.Index64(np.array([0], dtype=np.int64))
    stops = ak.index.Index64(np.array([4], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([10, 20, 30, 40], dtype=np.int64))
    arr = ak.Array(ak.contents.ListArray(starts, stops, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[10, 20, 30, 40]]


def test_to_cudf_listarray_overlapping_content():
    # Starts and stops can reference the same underlying content at different offsets.
    # After conversion to ListOffsetArray64 this must become non-overlapping.
    starts = ak.index.Index64(np.array([0, 2], dtype=np.int64))
    stops = ak.index.Index64(np.array([3, 5], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    arr = ak.Array(ak.contents.ListArray(starts, stops, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[1, 2, 3], [3, 4, 5]]


def test_to_cudf_listarray_float64():
    starts = ak.index.Index64(np.array([0, 2, 4], dtype=np.int64))
    stops = ak.index.Index64(np.array([2, 4, 4], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64))
    arr = ak.Array(ak.contents.ListArray(starts, stops, content))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == pytest.approx([1.1, 2.2])
    assert result[1] == pytest.approx([3.3, 4.4])
    assert result[2] == []


def test_to_cudf_listarray_roundtrip():
    starts = ak.index.Index64(np.array([0, 2, 2, 5], dtype=np.int64))
    stops = ak.index.Index64(np.array([2, 2, 5, 6], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([10, 20, 30, 40, 50, 60], dtype=np.int64))
    arr = ak.Array(ak.contents.ListArray(starts, stops, content))
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [[10, 20], [], [30, 40, 50], [60]]


# ---------------------------------------------------------------------------
# ByteMaskedArray — explicit byte-mask construction
# ---------------------------------------------------------------------------


def test_to_cudf_bytemasked_int32_valid_when_true():
    # valid_when=True: 1-bytes mark valid entries, 0-bytes mark nulls.
    mask = ak.index.Index8(np.array([1, 0, 1, 0, 1], dtype=np.int8))
    content = ak.contents.NumpyArray(np.array([10, 99, 30, 99, 50], dtype=np.int32))
    arr = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=True))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == 10
    assert result[1] is None
    assert result[2] == 30
    assert result[3] is None
    assert result[4] == 50


def test_to_cudf_bytemasked_int32_valid_when_false():
    # valid_when=False means 0-bytes denote valid entries and 1-bytes denote
    # nulls in the awkward array.  However, ByteMaskedArray._to_cudf packs the
    # raw byte values via packbits without inverting for valid_when — libcudf
    # always treats 1-bits as valid.  As a result the cuDF output reflects the
    # raw bit values: positions where mask==1 are valid in cuDF, and positions
    # where mask==0 appear as null, which is the opposite of the awkward
    # semantics.  This test documents that existing behavior.
    mask = ak.index.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8))
    content = ak.contents.NumpyArray(np.array([10, 99, 30, 99, 50], dtype=np.int32))
    arr = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=False))
    # Verify awkward interprets it correctly (0=valid when valid_when=False)
    assert ak.to_list(arr) == [10, None, 30, None, 50]
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    # _to_cudf packs raw bytes without inverting: 1-bits → valid in cuDF.
    # So cuDF sees positions with mask==1 as valid and mask==0 as null.
    assert result[0] is None  # mask=0 → 0-bit → null in cuDF
    assert result[1] == 99  # mask=1 → 1-bit → valid in cuDF (content value)
    assert result[2] is None
    assert result[3] == 99
    assert result[4] is None


def test_to_cudf_bytemasked_all_valid():
    mask = ak.index.Index8(np.array([1, 1, 1, 1], dtype=np.int8))
    content = ak.contents.NumpyArray(np.array([1, 2, 3, 4], dtype=np.int64))
    arr = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=True))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [1, 2, 3, 4]


def test_to_cudf_bytemasked_all_null():
    mask = ak.index.Index8(np.array([0, 0, 0], dtype=np.int8))
    content = ak.contents.NumpyArray(np.array([99, 99, 99], dtype=np.int64))
    arr = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=True))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [None, None, None]


def test_to_cudf_bytemasked_float64():
    mask = ak.index.Index8(np.array([1, 0, 1], dtype=np.int8))
    content = ak.contents.NumpyArray(np.array([1.5, 99.9, 3.5], dtype=np.float64))
    arr = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=True))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == pytest.approx(1.5)
    assert result[1] is None
    assert result[2] == pytest.approx(3.5)


def test_to_cudf_bytemasked_bool():
    mask = ak.index.Index8(np.array([1, 0, 1, 1], dtype=np.int8))
    content = ak.contents.NumpyArray(
        np.array([True, False, False, True], dtype=np.bool_)
    )
    arr = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=True))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] is True
    assert result[1] is None
    assert result[2] is False
    assert result[3] is True


def test_to_cudf_bytemasked_roundtrip():
    mask = ak.index.Index8(np.array([1, 0, 0, 1, 1], dtype=np.int8))
    content = ak.contents.NumpyArray(np.array([10, 99, 99, 40, 50], dtype=np.int32))
    arr = ak.Array(ak.contents.ByteMaskedArray(mask, content, valid_when=True))
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [10, None, None, 40, 50]


# ---------------------------------------------------------------------------
# UnmaskedArray — passthrough wrapper with no mask
# ---------------------------------------------------------------------------


def test_to_cudf_unmasked_int32():
    # UnmaskedArray wraps content without adding any mask.
    content = ak.contents.NumpyArray(np.array([1, 2, 3, 4], dtype=np.int32))
    arr = ak.Array(ak.contents.UnmaskedArray(content))
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert _to_arrow_list(out) == [1, 2, 3, 4]
    assert out.dtype == np.dtype("int32")


def test_to_cudf_unmasked_float64():
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3], dtype=np.float64))
    arr = ak.Array(ak.contents.UnmaskedArray(content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == pytest.approx([1.1, 2.2, 3.3])


def test_to_cudf_unmasked_bool():
    content = ak.contents.NumpyArray(np.array([True, False, True], dtype=np.bool_))
    arr = ak.Array(ak.contents.UnmaskedArray(content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [True, False, True]


def test_to_cudf_unmasked_empty():
    content = ak.contents.NumpyArray(np.array([], dtype=np.int64))
    arr = ak.Array(ak.contents.UnmaskedArray(content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == []
    assert out.dtype == np.dtype("int64")


def test_to_cudf_unmasked_roundtrip():
    content = ak.contents.NumpyArray(np.array([10, 20, 30], dtype=np.int64))
    arr = ak.Array(ak.contents.UnmaskedArray(content))
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [10, 20, 30]


# ---------------------------------------------------------------------------
# EmptyArray — zero-length fallback to float64
# ---------------------------------------------------------------------------


def test_to_cudf_emptyarray_zero_length():
    # EmptyArray has no dtype; _to_cudf converts it to a zero-length float64.
    arr = ak.Array(ak.contents.EmptyArray())
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert len(out) == 0


def test_to_cudf_emptyarray_as_list_content():
    # EmptyArray as the content of an empty list — the outer length is nonzero
    # but the flat content is empty.
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 0, 0], dtype=np.int64)),
            ak.contents.EmptyArray(),
        )
    )
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert _to_arrow_list(out) == [[], []]


def test_to_cudf_emptyarray_nested():
    # EmptyArray nested as content inside a ListOffsetArray with zero entries.
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0], dtype=np.int64)),
            ak.contents.EmptyArray(),
        )
    )
    out = ak.to_cudf(arr)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# IndexedArray — carry/project path
# ---------------------------------------------------------------------------


def test_to_cudf_indexedarray_basic():
    # IndexedArray re-orders / repeats elements via an index.
    index = ak.index.Index64(np.array([2, 0, 1, 2], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([10, 20, 30], dtype=np.int32))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [30, 10, 20, 30]


def test_to_cudf_indexedarray_identity():
    # Identity permutation — same order as content.
    index = ak.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([1, 2, 3, 4], dtype=np.int64))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [1, 2, 3, 4]


def test_to_cudf_indexedarray_reverse():
    # Reverse permutation.
    index = ak.index.Index64(np.array([3, 2, 1, 0], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([10, 20, 30, 40], dtype=np.int32))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [40, 30, 20, 10]


def test_to_cudf_indexedarray_repeated():
    # Index can repeat elements.
    index = ak.index.Index64(np.array([0, 0, 1, 1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([5, 6], dtype=np.int32))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [5, 5, 6, 6]


def test_to_cudf_indexedarray_float64():
    index = ak.index.Index64(np.array([1, 0, 2], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3], dtype=np.float64))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result == pytest.approx([2.2, 1.1, 3.3])


def test_to_cudf_indexedarray_empty_index():
    # Index with zero elements but non-empty content.
    index = ak.index.Index64(np.array([], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([10, 20, 30], dtype=np.int64))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == []


def test_to_cudf_indexedarray_empty_content():
    # Empty content — the index must also be empty for a valid IndexedArray.
    index = ak.index.Index64(np.array([], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([], dtype=np.int32))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == []


def test_to_cudf_indexedarray_roundtrip():
    index = ak.index.Index64(np.array([2, 0, 1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([100, 200, 300], dtype=np.int64))
    arr = ak.Array(ak.contents.IndexedArray(index, content))
    result = ak.from_cudf(ak.to_cudf(arr))
    assert ak.to_list(result) == [300, 100, 200]


# ---------------------------------------------------------------------------
# RegularArray — not implemented; must raise NotImplementedError
# ---------------------------------------------------------------------------


def test_to_cudf_regulararray_raises():
    # RegularArray has no _to_cudf override — must raise NotImplementedError.
    content = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))
    arr = ak.Array(ak.contents.RegularArray(content, size=2))
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


def test_to_cudf_regulararray_3d_raises():
    # 3-D regular (matrix-like) array.
    content = ak.contents.NumpyArray(np.arange(12, dtype=np.int32))
    inner = ak.contents.RegularArray(content, size=3)  # shape (4, 3)
    outer = ak.contents.RegularArray(inner, size=2)  # shape (2, 2, 3)
    arr = ak.Array(outer)
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


def test_to_cudf_regulararray_empty_raises():
    # Even an empty RegularArray must raise, not silently succeed.
    content = ak.contents.NumpyArray(np.array([], dtype=np.float64))
    arr = ak.Array(ak.contents.RegularArray(content, size=3, zeros_length=0))
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


def test_to_cudf_regulararray_size_1_raises():
    # Size-1 regular arrays could in principle be flattened, but no _to_cudf
    # override exists — must still raise.
    content = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64))
    arr = ak.Array(ak.contents.RegularArray(content, size=1))
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


# ---------------------------------------------------------------------------
# UnionArray — not implemented; must raise NotImplementedError
# ---------------------------------------------------------------------------


def test_to_cudf_unionarray_raises():
    # UnionArray has no _to_cudf override — must raise NotImplementedError.
    tags = ak.index.Index8(np.array([0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1], dtype=np.int64))
    content0 = ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32))
    content1 = ak.contents.NumpyArray(np.array([1.1, 2.2], dtype=np.float64))
    arr = ak.Array(ak.contents.UnionArray(tags, index, [content0, content1]))
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


def test_to_cudf_unionarray_int_and_bool_raises():
    tags = ak.index.Index8(np.array([0, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 1, 0, 1], dtype=np.int64))
    content0 = ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64))
    content1 = ak.contents.NumpyArray(np.array([True, False], dtype=np.bool_))
    arr = ak.Array(ak.contents.UnionArray(tags, index, [content0, content1]))
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


def test_to_cudf_unionarray_three_types_raises():
    tags = ak.index.Index8(np.array([0, 1, 2], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 0], dtype=np.int64))
    content0 = ak.contents.NumpyArray(np.array([1], dtype=np.int32))
    content1 = ak.contents.NumpyArray(np.array([1.5], dtype=np.float32))
    content2 = ak.contents.NumpyArray(np.array([True], dtype=np.bool_))
    arr = ak.Array(ak.contents.UnionArray(tags, index, [content0, content1, content2]))
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


def test_to_cudf_unionarray_nested_raises():
    # UnionArray whose contents are themselves list arrays.
    tags = ak.index.Index8(np.array([0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0], dtype=np.int64))
    content0 = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([1, 2], dtype=np.int32)),
    )
    content1 = ak.contents.NumpyArray(np.array([99.9], dtype=np.float64))
    arr = ak.Array(ak.contents.UnionArray(tags, index, [content0, content1]))
    with pytest.raises(NotImplementedError):
        ak.to_cudf(arr)


# ---------------------------------------------------------------------------
# BitMaskedArray — additional coverage (all-null, single-element, large)
# ---------------------------------------------------------------------------


def test_to_cudf_bitmasked_all_null_lsb():
    # All entries are masked as null (valid_when=True, lsb_order=True)
    # Force typed content via IndexedOptionArray then convert to BitMasked
    index = ak.index.Index64(np.array([-1, -1, -1, -1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([], dtype=np.int32))
    opt = ak.Array(ak.contents.IndexedOptionArray(index, content))
    bit_arr = ak.Array(opt.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(bit_arr)
    assert _to_arrow_list(out) == [None, None, None, None]


def test_to_cudf_bitmasked_single_valid():
    # Use an option-typed array so the layout has to_BitMaskedArray available.
    # ak.Array([42]) is a plain NumpyArray; ak.Array([42, None])[0:1] gives an
    # IndexedOptionArray which supports to_BitMaskedArray.
    opt = ak.Array([42, None])
    single = ak.Array(opt.layout[0:1])  # IndexedOptionArray of length 1
    bit_arr = ak.Array(single.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(bit_arr)
    assert _to_arrow_list(out) == [42]


def test_to_cudf_bitmasked_single_null():
    index = ak.index.Index64(np.array([-1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([], dtype=np.int64))
    opt = ak.Array(ak.contents.IndexedOptionArray(index, content))
    bit_arr = ak.Array(opt.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(bit_arr)
    assert _to_arrow_list(out) == [None]


def test_to_cudf_bitmasked_large_lsb():
    # 64+ elements to exercise multi-byte mask alignment.
    values = list(range(64))
    nullable_values = [v if v % 2 == 0 else None for v in values]
    arr = ak.Array(nullable_values)
    bit_arr = ak.Array(arr.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(bit_arr)
    assert _to_arrow_list(out) == nullable_values


def test_to_cudf_bitmasked_large_msb():
    # 64+ elements with msb_order=False (MSB-first packing).
    values = list(range(64))
    nullable_values = [v if v % 3 != 0 else None for v in values]
    arr = ak.Array(nullable_values)
    bit_arr = ak.Array(arr.layout.to_BitMaskedArray(True, False))
    out = ak.to_cudf(bit_arr)
    assert _to_arrow_list(out) == nullable_values


def test_to_cudf_bitmasked_float64():
    nullable = ak.Array([1.1, None, 3.3, None, 5.5])
    bit_arr = ak.Array(nullable.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(bit_arr)
    result = _to_arrow_list(out)
    assert result[0] == pytest.approx(1.1)
    assert result[1] is None
    assert result[2] == pytest.approx(3.3)
    assert result[3] is None
    assert result[4] == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# IndexedOptionArray — additional edge cases
# ---------------------------------------------------------------------------


def test_to_cudf_indexedoption_uint32():
    index = ak.index.Index64(np.array([0, -1, 2, -1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([10, 0, 30], dtype=np.uint32))
    arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == 10
    assert result[1] is None
    assert result[2] == 30
    assert result[3] is None


def test_to_cudf_indexedoption_uint64():
    index = ak.index.Index64(np.array([-1, 0], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([2**63], dtype=np.uint64))
    arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] is None
    assert result[1] == 2**63


def test_to_cudf_indexedoption_single_entry_valid():
    index = ak.index.Index64(np.array([0], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([42], dtype=np.int32))
    arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [42]


def test_to_cudf_indexedoption_single_entry_null():
    index = ak.index.Index64(np.array([-1], dtype=np.int64))
    content = ak.contents.NumpyArray(np.array([], dtype=np.int32))
    arr = ak.Array(ak.contents.IndexedOptionArray(index, content))
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [None]


def test_to_cudf_indexedoption_list_content():
    # Each outer entry may be None; the valid entries are lists.
    arr = ak.Array([[1, 2], None, [], None, [3]])
    out = ak.to_cudf(arr)
    assert _to_arrow_list(out) == [[1, 2], None, [], None, [3]]


def test_to_cudf_indexedoption_struct_content():
    # Each outer entry may be None; valid entries are structs.
    arr = ak.Array([{"x": 1}, None, {"x": 3}])
    out = ak.to_cudf(arr)
    result = _to_arrow_list(out)
    assert result[0] == {"x": 1}
    assert result[1] is None
    assert result[2] == {"x": 3}


# ---------------------------------------------------------------------------
# Datetime and timedelta dtypes (NumpyArray temporal path)
# ---------------------------------------------------------------------------


def test_to_cudf_datetime64_ns():
    dates = np.array(["2021-01-01", "2022-06-15", "2023-12-31"], dtype="datetime64[ns]")
    arr = ak.Array(ak.contents.NumpyArray(dates))
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert len(out) == 3


def test_to_cudf_datetime64_us():
    dates = np.array(["2020-01-01", "2020-07-04"], dtype="datetime64[us]")
    arr = ak.Array(ak.contents.NumpyArray(dates))
    out = ak.to_cudf(arr)
    assert out.dtype == np.dtype("datetime64[us]")
    assert len(out) == 2


def test_to_cudf_datetime64_ms():
    dates = np.array(["2000-01-01", "2000-06-15"], dtype="datetime64[ms]")
    arr = ak.Array(ak.contents.NumpyArray(dates))
    out = ak.to_cudf(arr)
    assert out.dtype == np.dtype("datetime64[ms]")


def test_to_cudf_datetime64_s():
    dates = np.array(["2010-05-01", "2010-05-02"], dtype="datetime64[s]")
    arr = ak.Array(ak.contents.NumpyArray(dates))
    out = ak.to_cudf(arr)
    assert out.dtype == np.dtype("datetime64[s]")


def test_to_cudf_timedelta64_ns():
    deltas = np.array([0, 1_000_000, 1_000_000_000], dtype="timedelta64[ns]")
    arr = ak.Array(ak.contents.NumpyArray(deltas))
    out = ak.to_cudf(arr)
    assert out.dtype == np.dtype("timedelta64[ns]")
    assert len(out) == 3


def test_to_cudf_timedelta64_us():
    deltas = np.array([0, 500, 1000], dtype="timedelta64[us]")
    arr = ak.Array(ak.contents.NumpyArray(deltas))
    out = ak.to_cudf(arr)
    assert out.dtype == np.dtype("timedelta64[us]")


def test_to_cudf_timedelta64_ms():
    deltas = np.array([0, 100, 200], dtype="timedelta64[ms]")
    arr = ak.Array(ak.contents.NumpyArray(deltas))
    out = ak.to_cudf(arr)
    assert out.dtype == np.dtype("timedelta64[ms]")


def test_to_cudf_timedelta64_s():
    deltas = np.array([0, 3600, 86400], dtype="timedelta64[s]")
    arr = ak.Array(ak.contents.NumpyArray(deltas))
    out = ak.to_cudf(arr)
    assert out.dtype == np.dtype("timedelta64[s]")
    assert len(out) == 3


def test_to_cudf_datetime_roundtrip():
    from packaging.version import parse as parse_version

    cudf_ver = parse_version(cudf.__version__)
    if cudf_ver < parse_version("25.02.00"):
        pytest.skip("ak.from_cudf requires cudf >= 25.02")
    dates = np.array(["2021-01-01", "2022-06-15", "2023-12-31"], dtype="datetime64[ms]")
    arr = ak.Array(ak.contents.NumpyArray(dates))
    result = ak.from_cudf(ak.to_cudf(arr))
    assert result.layout.data.dtype == cupy.dtype("datetime64[ms]")
    assert len(result) == 3
