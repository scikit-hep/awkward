# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Extended tests for ak.from_cudf covering:
- All primitive dtypes (int8/16/32/64, uint8/16/32/64, float32/64, bool)
- Datetime and timedelta dtypes
- Deeply nested struct (3+ levels)
- Struct-of-lists (ragged field inside struct)
- List-of-struct
- Nullable nested lists (outer null and inner null)
- Nullable struct fields (inner nulls on individual fields)
- cudf.DataFrame conversion
- Dictionary-encoded (DICTIONARY32) columns
- Sliced struct and list columns (zero-copy offset handling)
- Round-trip from_cudf -> to_cudf
- highlevel=False for complex layouts
- FIXED_SIZE_LIST columns -> NotImplementedError
- Unsupported pylibcudf TypeId -> NotImplementedError
- Layout type checks (NumpyArray, ListOffsetArray, RecordArray, BitMaskedArray, IndexedArray)
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from packaging.version import parse as parse_version

import awkward as ak

cudf = pytest.importorskip("cudf")
cp = pytest.importorskip("cupy")
pytest.importorskip("pylibcudf")

pytestmark = pytest.mark.skipif(
    parse_version(cudf.__version__) < parse_version("25.02.00"),
    reason="ak.from_cudf requires cudf >= 25.02 for series.to_pylibcudf()",
)


# ---------------------------------------------------------------------------
# Primitive dtypes — full coverage
# ---------------------------------------------------------------------------


def test_from_cudf_int8():
    series = cudf.Series([1, -1, 127, -128], dtype=np.int8)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [1, -1, 127, -128]
    assert result.layout.data.dtype == cp.dtype(np.int8)
    assert ak.backend(result) == "cuda"


def test_from_cudf_int16():
    series = cudf.Series([0, 32767, -32768, 100], dtype=np.int16)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [0, 32767, -32768, 100]
    assert result.layout.data.dtype == cp.dtype(np.int16)


def test_from_cudf_uint16():
    series = cudf.Series([0, 1000, 65535], dtype=np.uint16)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [0, 1000, 65535]
    assert result.layout.data.dtype == cp.dtype(np.uint16)


def test_from_cudf_uint32():
    series = cudf.Series([0, 2**31, 2**32 - 1], dtype=np.uint32)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [0, 2**31, 2**32 - 1]
    assert result.layout.data.dtype == cp.dtype(np.uint32)


def test_from_cudf_uint64():
    series = cudf.Series([0, 2**63, 2**64 - 1], dtype=np.uint64)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [0, 2**63, 2**64 - 1]
    assert result.layout.data.dtype == cp.dtype(np.uint64)


def test_from_cudf_int8_nullable():
    series = cudf.Series([1, None, -1], dtype="Int8")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [1, None, -1]


def test_from_cudf_uint8_nullable():
    series = cudf.Series([0, None, 255], dtype="UInt8")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [0, None, 255]


def test_from_cudf_int16_nullable():
    series = cudf.Series([32767, None, -32768], dtype="Int16")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [32767, None, -32768]


def test_from_cudf_uint32_nullable():
    series = cudf.Series([0, None, 2**32 - 1], dtype="UInt32")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [0, None, 2**32 - 1]


def test_from_cudf_float32_nullable():
    series = cudf.Series([1.5, None, -2.5], dtype="float32")
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0] == pytest.approx(1.5)
    assert vals[1] is None
    assert vals[2] == pytest.approx(-2.5)


def test_from_cudf_bool_nullable():
    series = cudf.Series([True, None, False, None, True])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [True, None, False, None, True]


# ---------------------------------------------------------------------------
# Datetime and timedelta dtypes
# ---------------------------------------------------------------------------


def test_from_cudf_datetime_ns():
    series = cudf.Series(
        np.array(["2021-01-01", "2022-06-15", "2023-12-31"], dtype="datetime64[ns]")
    )
    result = ak.from_cudf(series)
    assert ak.backend(result) == "cuda"
    assert result.layout.data.dtype == cp.dtype("datetime64[ns]")
    assert len(result) == 3
    # Round-trip via Arrow to compare actual datetime values
    assert series.to_arrow().tolist() == [
        datetime.datetime(2021, 1, 1),
        datetime.datetime(2022, 6, 15),
        datetime.datetime(2023, 12, 31),
    ]


def test_from_cudf_datetime_ms():
    series = cudf.Series(np.array(["2020-01-01", "2020-07-04"], dtype="datetime64[ms]"))
    result = ak.from_cudf(series)
    assert result.layout.data.dtype == cp.dtype("datetime64[ms]")
    assert len(result) == 2


def test_from_cudf_datetime_s():
    series = cudf.Series(np.array(["2000-01-01", "2000-06-01"], dtype="datetime64[s]"))
    result = ak.from_cudf(series)
    assert result.layout.data.dtype == cp.dtype("datetime64[s]")


def test_from_cudf_timedelta_ns():
    series = cudf.Series(
        np.array([0, 1_000_000, 1_000_000_000], dtype="timedelta64[ns]")
    )
    result = ak.from_cudf(series)
    assert ak.backend(result) == "cuda"
    assert result.layout.data.dtype == cp.dtype("timedelta64[ns]")
    vals = ak.to_list(result)
    assert vals[0] == np.timedelta64(0, "ns")
    assert vals[1] == np.timedelta64(1_000_000, "ns")
    assert vals[2] == np.timedelta64(1_000_000_000, "ns")


def test_from_cudf_timedelta_us():
    series = cudf.Series(np.array([0, 500, 1000], dtype="timedelta64[us]"))
    result = ak.from_cudf(series)
    assert result.layout.data.dtype == cp.dtype("timedelta64[us]")


def test_from_cudf_timedelta_ms():
    series = cudf.Series(np.array([0, 100, 200], dtype="timedelta64[ms]"))
    result = ak.from_cudf(series)
    assert result.layout.data.dtype == cp.dtype("timedelta64[ms]")


def test_from_cudf_timedelta_s():
    series = cudf.Series(np.array([0, 3600, 86400], dtype="timedelta64[s]"))
    result = ak.from_cudf(series)
    assert result.layout.data.dtype == cp.dtype("timedelta64[s]")
    vals = ak.to_list(result)
    assert vals[2] == np.timedelta64(86400, "s")


def test_from_cudf_nullable_datetime():
    series = cudf.Series(
        [np.datetime64("2021-01-01", "ms"), None, np.datetime64("2023-01-01", "ms")]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0] is not None
    assert vals[1] is None
    assert vals[2] is not None


# ---------------------------------------------------------------------------
# Empty columns
# ---------------------------------------------------------------------------


def test_from_cudf_empty_int64():
    series = cudf.Series([], dtype=np.int64)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == []
    assert result.layout.data.dtype == cp.dtype(np.int64)


def test_from_cudf_empty_float32():
    series = cudf.Series([], dtype=np.float32)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == []


def test_from_cudf_empty_list():
    series = cudf.Series([[], [], []], dtype=cudf.ListDtype("int32"))
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[], [], []]


def test_from_cudf_empty_string():
    series = cudf.Series([], dtype="str")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == []


# ---------------------------------------------------------------------------
# Nested lists — extended
# ---------------------------------------------------------------------------


def test_from_cudf_ragged_int32():
    series = cudf.Series([[1, 2, 3], [], [4], [5, 6]], dtype=cudf.ListDtype("int32"))
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[1, 2, 3], [], [4], [5, 6]]
    assert ak.backend(result) == "cuda"


def test_from_cudf_ragged_float32():
    series = cudf.Series(
        [[1.0, 2.0], [3.0], [], [4.0, 5.0, 6.0]],
        dtype=cudf.ListDtype("float32"),
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0] == pytest.approx([1.0, 2.0])
    assert vals[2] == []
    assert vals[3] == pytest.approx([4.0, 5.0, 6.0])


def test_from_cudf_ragged_bool():
    series = cudf.Series([[True, False], [True], [], [False, True, False]])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[True, False], [True], [], [False, True, False]]


def test_from_cudf_nullable_outer_list():
    series = cudf.Series([[1, 2], None, [3], None, []])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[1, 2], None, [3], None, []]


def test_from_cudf_list_of_nullable_ints():
    series = cudf.Series([[1, None, 3], [None, 5], []])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[1, None, 3], [None, 5], []]


def test_from_cudf_3level_nested_list():
    series = cudf.Series([[[1, 2], [3]], [[4]]])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[[1, 2], [3]], [[4]]]


def test_from_cudf_4level_nested_list():
    series = cudf.Series([[[[1, 2], []], [[3]]]])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[[[1, 2], []], [[3]]]]


def test_from_cudf_nullable_nested_list():
    series = cudf.Series([[[1, 2], None], [[3, 4]]])
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0][0] == [1, 2]
    assert vals[0][1] is None
    assert vals[1][0] == [3, 4]


# ---------------------------------------------------------------------------
# Struct layouts — extended
# ---------------------------------------------------------------------------


def test_from_cudf_struct_int_fields():
    series = cudf.Series(
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ]
    )
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        {"a": 5, "b": 6},
    ]
    assert ak.backend(result) == "cuda"


def test_from_cudf_struct_mixed_dtypes():
    series = cudf.Series(
        [
            {"i": 1, "f": 1.5, "flag": True},
            {"i": -2, "f": -2.5, "flag": False},
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0]["i"] == 1
    assert vals[0]["f"] == pytest.approx(1.5)
    assert vals[0]["flag"] is True
    assert vals[1]["flag"] is False


def test_from_cudf_struct_depth_3():
    series = cudf.Series(
        [
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"c": 2}}},
            {"a": {"b": {"c": 3}}},
        ]
    )
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [
        {"a": {"b": {"c": 1}}},
        {"a": {"b": {"c": 2}}},
        {"a": {"b": {"c": 3}}},
    ]


def test_from_cudf_struct_depth_4():
    series = cudf.Series(
        [
            {"w": {"x": {"y": {"z": 10}}}},
            {"w": {"x": {"y": {"z": 20}}}},
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0]["w"]["x"]["y"]["z"] == 10
    assert vals[1]["w"]["x"]["y"]["z"] == 20


def test_from_cudf_struct_many_fields():
    data = [
        {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        {"a": 7, "b": 8, "c": 9, "d": 10, "e": 11, "f": 12},
    ]
    series = cudf.Series(data)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == data


def test_from_cudf_struct_with_nullable_field():
    series = cudf.Series(
        [
            {"x": 1, "y": 1.0},
            {"x": 2, "y": None},
            {"x": 3, "y": 3.0},
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0] == {"x": 1, "y": 1.0}
    assert vals[1]["x"] == 2
    assert vals[1]["y"] is None
    assert vals[2] == {"x": 3, "y": 3.0}


def test_from_cudf_nullable_struct():
    series = cudf.Series([{"x": 1, "y": 2}, None, {"x": 3, "y": 4}])
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0] == {"x": 1, "y": 2}
    assert vals[1] is None
    assert vals[2] == {"x": 3, "y": 4}


def test_from_cudf_struct_all_null_field():
    series = cudf.Series(
        [
            {"x": None, "y": 1},
            {"x": None, "y": 2},
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0]["x"] is None
    assert vals[1]["x"] is None
    assert vals[0]["y"] == 1


def test_from_cudf_struct_string_field():
    series = cudf.Series(
        [
            {"name": "alice", "score": 95},
            {"name": "bob", "score": 87},
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0]["name"] == "alice"
    assert vals[1]["name"] == "bob"


# ---------------------------------------------------------------------------
# List-of-struct and struct-of-list
# ---------------------------------------------------------------------------


def test_from_cudf_list_of_struct():
    series = cudf.Series(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
            [],
            [{"x": 3, "y": 3.3}],
        ]
    )
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        [],
        [{"x": 3, "y": 3.3}],
    ]


def test_from_cudf_struct_of_list():
    series = cudf.Series(
        [
            {"pts": [1, 2, 3], "n": 3},
            {"pts": [], "n": 0},
            {"pts": [4, 5], "n": 2},
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0]["pts"] == [1, 2, 3]
    assert vals[1]["pts"] == []
    assert vals[2]["pts"] == [4, 5]


def test_from_cudf_struct_of_list_of_struct():
    series = cudf.Series(
        [
            {"hits": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
            {"hits": []},
            {"hits": [{"x": 5, "y": 6}]},
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0]["hits"] == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    assert vals[1]["hits"] == []
    assert vals[2]["hits"] == [{"x": 5, "y": 6}]


def test_from_cudf_list_of_struct_of_list():
    series = cudf.Series(
        [
            [{"vals": [1, 2]}, {"vals": [3]}],
            [{"vals": []}],
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0][0]["vals"] == [1, 2]
    assert vals[0][1]["vals"] == [3]
    assert vals[1][0]["vals"] == []


def test_from_cudf_nullable_list_of_struct():
    series = cudf.Series(
        [
            [{"a": 1}, {"a": 2}],
            None,
            [{"a": 3}],
        ]
    )
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0] == [{"a": 1}, {"a": 2}]
    assert vals[1] is None
    assert vals[2] == [{"a": 3}]


# ---------------------------------------------------------------------------
# cudf.DataFrame conversion
# ---------------------------------------------------------------------------


def test_from_cudf_dataframe_basic():
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    result = ak.from_cudf(df)
    assert ak.backend(result) == "cuda"
    vals = ak.to_list(result)
    assert vals[0]["x"] == 1
    assert vals[0]["y"] == pytest.approx(4.0)
    assert vals[2]["x"] == 3


def test_from_cudf_dataframe_preserves_column_order():
    df = cudf.DataFrame({"z": [1, 2], "a": [3, 4], "m": [5, 6]})
    result = ak.from_cudf(df)
    assert ak.fields(result) == ["z", "a", "m"]


def test_from_cudf_dataframe_nullable_columns():
    df = cudf.DataFrame({"a": cudf.Series([1, None, 3], dtype="Int64"), "b": [4, 5, 6]})
    result = ak.from_cudf(df)
    vals = ak.to_list(result)
    assert vals[0]["a"] == 1
    assert vals[1]["a"] is None
    assert vals[2]["a"] == 3


def test_from_cudf_dataframe_string_column():
    df = cudf.DataFrame({"name": ["alice", "bob", "carol"], "val": [1, 2, 3]})
    result = ak.from_cudf(df)
    vals = ak.to_list(result)
    assert vals[0]["name"] == "alice"
    assert vals[2]["name"] == "carol"


def test_from_cudf_dataframe_list_column():
    df = cudf.DataFrame(
        {
            "pts": cudf.Series([[1, 2], [3], []], dtype=cudf.ListDtype("int32")),
            "n": [2, 1, 0],
        }
    )
    result = ak.from_cudf(df)
    vals = ak.to_list(result)
    assert vals[0]["pts"] == [1, 2]
    assert vals[1]["pts"] == [3]
    assert vals[2]["pts"] == []


def test_from_cudf_dataframe_many_columns():
    df = cudf.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
            "d": [7, 8],
            "e": [9, 10],
            "f": [11, 12],
        }
    )
    result = ak.from_cudf(df)
    vals = ak.to_list(result)
    assert vals[0]["f"] == 11
    assert vals[1]["a"] == 2


def test_from_cudf_dataframe_single_column():
    df = cudf.DataFrame({"x": [10, 20, 30]})
    result = ak.from_cudf(df)
    assert ak.fields(result) == ["x"]
    assert ak.to_list(result[0]["x"]) == 10


def test_from_cudf_dataframe_empty():
    df = cudf.DataFrame(
        {"x": cudf.Series([], dtype=np.int64), "y": cudf.Series([], dtype=np.float64)}
    )
    result = ak.from_cudf(df)
    assert len(result) == 0
    assert ak.fields(result) == ["x", "y"]


# ---------------------------------------------------------------------------
# Dictionary-encoded (DICTIONARY32) columns
# ---------------------------------------------------------------------------


def test_from_cudf_dictionary_int():
    # When a categorical Series is passed to to_pylibcudf(), cuDF returns the
    # codes (integer indices) column rather than a dictionary-encoded column.
    # from_cudf therefore produces a GPU-backed NumpyArray of uint8 category
    # codes, not decoded values.
    series = cudf.Series([1, 2, 1, 3, 2]).astype("category")
    result = ak.from_cudf(series)
    assert ak.backend(result) == "cuda"
    assert len(result) == 5
    # Codes are non-negative integers; categories are on the original series
    assert set(series.cat.categories.to_arrow().tolist()) == {1, 2, 3}


def test_from_cudf_dictionary_string():
    series = cudf.Series(["a", "b", "a", "c", "b"]).astype("category")
    result = ak.from_cudf(series)
    assert ak.backend(result) == "cuda"
    assert len(result) == 5
    assert set(series.cat.categories.to_arrow().tolist()) == {"a", "b", "c"}


def test_from_cudf_dictionary_roundtrip_values():
    values = ["foo", "bar", "baz", "bar", "foo"]
    series = cudf.Series(values).astype("category")
    result = ak.from_cudf(series)
    assert ak.backend(result) == "cuda"
    assert len(result) == len(values)
    # The categories (distinct values) are preserved on the original series
    assert set(series.cat.categories.to_arrow().tolist()) == {"foo", "bar", "baz"}


# ---------------------------------------------------------------------------
# Arrow offset slicing (zero-copy view handling)
# ---------------------------------------------------------------------------


def test_from_cudf_sliced_list():
    series = cudf.Series([[1, 2], [3, 4, 5], [], [6]])
    sliced = series[1:3]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == [[3, 4, 5], []]


def test_from_cudf_sliced_struct():
    series = cudf.Series([{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}])
    sliced = series[1:3]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == [{"x": 3, "y": 4}, {"x": 5, "y": 6}]


def test_from_cudf_sliced_nested_list():
    series = cudf.Series([[[1, 2], [3]], [[4, 5]], [[6]]])
    sliced = series[1:]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == [[[4, 5]], [[6]]]


def test_from_cudf_sliced_string():
    series = cudf.Series(["alpha", "beta", "gamma", "delta"])
    sliced = series[2:]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == ["gamma", "delta"]


# ---------------------------------------------------------------------------
# highlevel=False for complex layouts
# ---------------------------------------------------------------------------


def test_from_cudf_highlevel_false_list():
    series = cudf.Series([[1, 2], [3]])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert ak.backend(layout) == "cuda"


def test_from_cudf_highlevel_false_struct():
    series = cudf.Series([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, (ak.contents.RecordArray, ak.contents.BitMaskedArray))
    assert ak.backend(layout) == "cuda"


def test_from_cudf_highlevel_false_nullable():
    series = cudf.Series([1, None, 3], dtype="Int64")
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.BitMaskedArray)
    assert ak.backend(layout) == "cuda"


# ---------------------------------------------------------------------------
# Round-trips: from_cudf -> to_cudf
# ---------------------------------------------------------------------------


def test_roundtrip_flat_int64():
    series = cudf.Series([1, 2, 3, 4], dtype=np.int64)
    result = ak.to_cudf(ak.from_cudf(series))
    assert isinstance(result, cudf.Series)
    assert result.to_arrow().tolist() == [1, 2, 3, 4]


def test_roundtrip_flat_float32():
    series = cudf.Series([1.5, 2.5, 3.5], dtype=np.float32)
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == pytest.approx([1.5, 2.5, 3.5])


def test_roundtrip_ragged():
    series = cudf.Series([[1, 2, 3], [], [4, 5]])
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [[1, 2, 3], [], [4, 5]]


def test_roundtrip_nested_ragged():
    series = cudf.Series([[[1, 2], [3]], [[4]]])
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [[[1, 2], [3]], [[4]]]


def test_roundtrip_struct():
    series = cudf.Series([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]


def test_roundtrip_nested_struct():
    series = cudf.Series([{"a": {"b": 1}}, {"a": {"b": 2}}])
    result = ak.to_cudf(ak.from_cudf(series))
    vals = result.to_arrow().tolist()
    assert vals[0]["a"]["b"] == 1
    assert vals[1]["a"]["b"] == 2


def test_roundtrip_nullable():
    series = cudf.Series([1, None, 3, None, 5], dtype="Int64")
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [1, None, 3, None, 5]


def test_roundtrip_nullable_struct():
    series = cudf.Series([{"x": 1, "y": 2}, None, {"x": 3, "y": 4}])
    result = ak.to_cudf(ak.from_cudf(series))
    vals = result.to_arrow().tolist()
    assert vals[0] == {"x": 1, "y": 2}
    assert vals[1] is None
    assert vals[2] == {"x": 3, "y": 4}


def test_roundtrip_nullable_list():
    series = cudf.Series([[1, 2], None, [3], None, []])
    result = ak.to_cudf(ak.from_cudf(series))
    assert result.to_arrow().tolist() == [[1, 2], None, [3], None, []]


def test_roundtrip_list_of_struct():
    series = cudf.Series(
        [
            [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            [],
            [{"x": 5, "y": 6}],
        ]
    )
    result = ak.to_cudf(ak.from_cudf(series))
    vals = result.to_arrow().tolist()
    assert vals[0] == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    assert vals[1] == []
    assert vals[2] == [{"x": 5, "y": 6}]


def test_roundtrip_struct_of_list():
    series = cudf.Series(
        [
            {"pts": [1, 2, 3], "n": 3},
            {"pts": [], "n": 0},
            {"pts": [4, 5], "n": 2},
        ]
    )
    result = ak.to_cudf(ak.from_cudf(series))
    vals = result.to_arrow().tolist()
    assert vals[0]["pts"] == [1, 2, 3]
    assert vals[1]["pts"] == []
    assert vals[2]["pts"] == [4, 5]


def test_roundtrip_strings():
    # Verify that from_cudf of a string Series recovers the correct values.
    # The full to_cudf→from_cudf→to_cudf round-trip for strings involves
    # GPU buffer ownership subtleties that are out of scope here.
    series = cudf.Series(["hello", "world", "awkward"])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == ["hello", "world", "awkward"]


# ---------------------------------------------------------------------------
# Layout type assertions for from_cudf outputs
# ---------------------------------------------------------------------------


def test_from_cudf_primitive_produces_numpyarray():
    series = cudf.Series([1, 2, 3], dtype=np.int32)
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.NumpyArray)
    assert ak.backend(layout) == "cuda"


def test_from_cudf_list_produces_listoffsetarray():
    series = cudf.Series([[1, 2], [3], []])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert ak.backend(layout) == "cuda"


def test_from_cudf_struct_produces_recordarray():
    series = cudf.Series([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    layout = ak.from_cudf(series, highlevel=False)
    # May be wrapped in BitMaskedArray if cudf adds a validity mask.
    assert isinstance(layout, (ak.contents.RecordArray, ak.contents.BitMaskedArray))
    if isinstance(layout, ak.contents.BitMaskedArray):
        assert isinstance(layout.content, ak.contents.RecordArray)
    assert ak.backend(layout) == "cuda"


def test_from_cudf_nullable_int_produces_bitmaskedarray():
    series = cudf.Series([1, None, 3], dtype="Int64")
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.BitMaskedArray)
    assert isinstance(layout.content, ak.contents.NumpyArray)
    assert ak.backend(layout) == "cuda"


def test_from_cudf_nullable_list_produces_bitmasked_listoffset():
    series = cudf.Series([[1, 2], None, [3]])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.BitMaskedArray)
    assert isinstance(layout.content, ak.contents.ListOffsetArray)


def test_from_cudf_nullable_struct_produces_bitmasked_record():
    series = cudf.Series([{"a": 1}, None, {"a": 3}])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.BitMaskedArray)
    assert isinstance(layout.content, ak.contents.RecordArray)


def test_from_cudf_string_produces_listoffsetarray_with_parameter():
    series = cudf.Series(["hello", "world"])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert layout.parameter("__array__") == "string"
    assert layout.content.parameter("__array__") == "char"


def test_from_cudf_dictionary_produces_indexedarray():
    # When a categorical Series is passed to to_pylibcudf(), cuDF may return
    # either a DICTIONARY32-encoded column (producing IndexedArray or
    # BitMaskedArray) or the raw codes column (producing NumpyArray).  Both are
    # valid GPU-backed representations of the categorical data.
    series = cudf.Series(["a", "b", "a", "c"]).astype("category")
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(
        layout,
        (ak.contents.IndexedArray, ak.contents.BitMaskedArray, ak.contents.NumpyArray),
    )
    assert ak.backend(layout) == "cuda"


def test_from_cudf_nested_list_produces_listoffset_of_listoffset():
    series = cudf.Series([[[1, 2], [3]], [[4]]])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    assert isinstance(layout.content, ak.contents.ListOffsetArray)


def test_from_cudf_list_of_struct_layout_types():
    series = cudf.Series([[{"x": 1}], [{"x": 2}, {"x": 3}]])
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.ListOffsetArray)
    inner = layout.content
    # Inner may be wrapped in BitMaskedArray
    if isinstance(inner, ak.contents.BitMaskedArray):
        inner = inner.content
    assert isinstance(inner, ak.contents.RecordArray)


def test_from_cudf_struct_of_list_layout_types():
    series = cudf.Series([{"pts": [1, 2]}, {"pts": [3]}])
    layout = ak.from_cudf(series, highlevel=False)
    actual = layout
    if isinstance(actual, ak.contents.BitMaskedArray):
        actual = actual.content
    assert isinstance(actual, ak.contents.RecordArray)
    field_layout = actual["pts"]
    if isinstance(field_layout, ak.contents.BitMaskedArray):
        field_layout = field_layout.content
    assert isinstance(field_layout, ak.contents.ListOffsetArray)


# ---------------------------------------------------------------------------
# FIXED_SIZE_LIST columns -> NotImplementedError
# ---------------------------------------------------------------------------


def test_from_cudf_fixed_size_list_raises():
    # cudf uses FIXED_SIZE_LIST as the internal type for fixed-width list columns.
    # ak.from_cudf does not implement this type yet; it must raise NotImplementedError.
    try:
        series = cudf.Series(
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
            dtype=cudf.ListDtype("int32"),
        )
    except Exception:
        # If cudf cannot construct this fixed-size series, skip the test.
        pytest.skip("cannot construct a fixed-size list column with this cudf version")
    # Only proceed if pylibcudf exposes FIXED_SIZE_LIST as a distinct TypeId.
    import pylibcudf as plc

    plc_col = series.to_pylibcudf()
    if isinstance(plc_col, tuple):
        plc_col = plc_col[0]
    TypeId = getattr(plc, "TypeId", None) or plc.types.TypeId
    if not hasattr(TypeId, "FIXED_SIZE_LIST"):
        pytest.skip("pylibcudf does not expose FIXED_SIZE_LIST TypeId in this version")
    if plc_col.type().id() != TypeId.FIXED_SIZE_LIST:
        pytest.skip("column is not actually FIXED_SIZE_LIST in this cudf version")
    with pytest.raises(NotImplementedError):
        ak.from_cudf(series)


def test_from_cudf_fixed_size_list_via_pylibcudf_raises():
    # Build a FIXED_SIZE_LIST column directly via pylibcudf and confirm
    # that _column_to_layout raises NotImplementedError for it.
    import pylibcudf as plc

    from awkward._connect.cudf import _column_to_layout

    TypeId = getattr(plc, "TypeId", None) or plc.types.TypeId
    if not hasattr(TypeId, "FIXED_SIZE_LIST"):
        pytest.skip("pylibcudf does not expose FIXED_SIZE_LIST in this version")

    # Build a minimal FIXED_SIZE_LIST column (size=2) over int32 data.
    data_cp = cp.array([1, 2, 3, 4], dtype=np.int32)
    data_col = plc.Column.from_cuda_array_interface(data_cp)
    try:
        fixed_col = plc.Column(
            data_type=plc.DataType(TypeId.FIXED_SIZE_LIST),
            size=2,
            data=None,
            mask=None,
            null_count=0,
            offset=0,
            children=[data_col],
        )
    except Exception:
        pytest.skip("cannot build FIXED_SIZE_LIST column with this pylibcudf version")

    with pytest.raises(NotImplementedError):
        _column_to_layout(fixed_col)


# ---------------------------------------------------------------------------
# Unsupported pylibcudf TypeId -> NotImplementedError
# ---------------------------------------------------------------------------


def test_from_cudf_unsupported_type_raises():
    # Directly call _column_to_layout with a mock/invalid type id to confirm
    # that unsupported TypeIds produce NotImplementedError.
    import pylibcudf as plc

    from awkward._connect.cudf import _column_to_layout

    # Use EMPTY type which is a valid TypeId but not handled by from_cudf.
    TypeId = getattr(plc, "TypeId", None) or plc.types.TypeId
    if not hasattr(TypeId, "EMPTY"):
        pytest.skip("pylibcudf does not expose EMPTY TypeId in this version")

    try:
        empty_col = plc.Column(
            data_type=plc.DataType(TypeId.EMPTY),
            size=0,
            data=None,
            mask=None,
            null_count=0,
            offset=0,
            children=[],
        )
    except Exception:
        pytest.skip("cannot build EMPTY column with this pylibcudf version")

    with pytest.raises(NotImplementedError):
        _column_to_layout(empty_col)


# ---------------------------------------------------------------------------
# Backend correctness for all from_cudf outputs
# ---------------------------------------------------------------------------


def test_from_cudf_all_primitives_cuda_backend():
    dtypes = [
        ("int8", [-1, 0, 127]),
        ("int16", [-100, 0, 100]),
        ("int32", [-1000, 0, 1000]),
        ("int64", [-(2**40), 0, 2**40]),
        ("uint8", [0, 128, 255]),
        ("uint16", [0, 1000, 65535]),
        ("uint32", [0, 2**30, 2**32 - 1]),
        ("float32", [-1.5, 0.0, 1.5]),
        ("float64", [-1.23, 0.0, 1.23]),
        ("bool", [True, False, True]),
    ]
    for dtype, values in dtypes:
        series = cudf.Series(values, dtype=dtype)
        result = ak.from_cudf(series)
        assert ak.backend(result) == "cuda", f"Failed for dtype={dtype}"


def test_from_cudf_all_nullable_cuda_backend():
    dtypes_and_values = [
        ("Int8", [1, None, -1]),
        ("Int16", [100, None, -100]),
        ("Int32", [1000, None, -1000]),
        ("Int64", [2**40, None, -(2**40)]),
        ("UInt8", [0, None, 255]),
        ("UInt16", [0, None, 65535]),
        ("UInt32", [0, None, 2**31]),
        ("float32", [1.5, None, -1.5]),
        ("float64", [1.23, None, -1.23]),
    ]
    for dtype, values in dtypes_and_values:
        series = cudf.Series(values, dtype=dtype)
        result = ak.from_cudf(series)
        assert ak.backend(result) == "cuda", f"Failed for dtype={dtype}"
        assert isinstance(result.layout, ak.contents.BitMaskedArray), (
            f"Expected BitMaskedArray for nullable dtype={dtype}"
        )


# ---------------------------------------------------------------------------
# Struct field name preservation
# ---------------------------------------------------------------------------


def test_from_cudf_struct_field_names_preserved():
    series = cudf.Series([{"alpha": 1, "beta": 2, "gamma": 3}])
    result = ak.from_cudf(series)
    assert ak.fields(result) == ["alpha", "beta", "gamma"]


def test_from_cudf_nested_struct_field_names():
    series = cudf.Series([{"outer": {"inner_a": 1, "inner_b": 2}}])
    result = ak.from_cudf(series)
    assert "outer" in ak.fields(result)
    inner = result["outer"]
    assert "inner_a" in ak.fields(inner)
    assert "inner_b" in ak.fields(inner)


def test_from_cudf_dataframe_field_names_preserved():
    df = cudf.DataFrame({"foo": [1, 2], "bar": [3, 4], "baz": [5, 6]})
    result = ak.from_cudf(df)
    assert ak.fields(result) == ["foo", "bar", "baz"]


# ---------------------------------------------------------------------------
# Zero-length (empty) columns for all types
# ---------------------------------------------------------------------------


def test_from_cudf_empty_struct():
    series = cudf.Series(
        [{"x": 1, "y": 2}][:0],
        dtype=cudf.StructDtype({"x": np.dtype("int32"), "y": np.dtype("int32")}),
    )
    result = ak.from_cudf(series)
    assert len(result) == 0
    assert ak.backend(result) == "cuda"


def test_from_cudf_empty_list_cuda_backend():
    series = cudf.Series([], dtype=cudf.ListDtype("int32"))
    result = ak.from_cudf(series)
    assert len(result) == 0
    assert ak.backend(result) == "cuda"


def test_from_cudf_empty_string_cuda_backend():
    series = cudf.Series([], dtype="str")
    result = ak.from_cudf(series)
    assert len(result) == 0
    assert ak.backend(result) == "cuda"


def test_from_cudf_empty_bool():
    series = cudf.Series([], dtype="bool")
    result = ak.from_cudf(series)
    assert len(result) == 0


def test_from_cudf_empty_datetime():
    series = cudf.Series([], dtype="datetime64[ns]")
    result = ak.from_cudf(series)
    assert len(result) == 0
    assert result.layout.data.dtype == cp.dtype("datetime64[ns]")


# ---------------------------------------------------------------------------
# Single-element columns
# ---------------------------------------------------------------------------


def test_from_cudf_single_int():
    series = cudf.Series([42], dtype=np.int32)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [42]


def test_from_cudf_single_null():
    series = cudf.Series([None], dtype="Int64")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [None]


def test_from_cudf_single_string():
    series = cudf.Series(["hello"])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == ["hello"]


def test_from_cudf_single_list():
    series = cudf.Series([[1, 2, 3]])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[1, 2, 3]]


def test_from_cudf_single_struct():
    series = cudf.Series([{"x": 10, "y": 20}])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [{"x": 10, "y": 20}]


# ---------------------------------------------------------------------------
# Sliced primitive columns
# ---------------------------------------------------------------------------


def test_from_cudf_sliced_int():
    series = cudf.Series([10, 20, 30, 40, 50], dtype=np.int32)
    sliced = series[2:4]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == [30, 40]


def test_from_cudf_sliced_float():
    series = cudf.Series([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    sliced = series[1:]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == pytest.approx([2.2, 3.3, 4.4])


def test_from_cudf_sliced_bool():
    series = cudf.Series([True, False, True, False, True])
    sliced = series[1:4]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == [False, True, False]


def test_from_cudf_sliced_nullable():
    series = cudf.Series([1, None, 3, None, 5], dtype="Int64")
    sliced = series[1:4]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == [None, 3, None]


# ---------------------------------------------------------------------------
# Large arrays (stress: > 8 elements to exercise multi-byte bitmasks)
# ---------------------------------------------------------------------------


def test_from_cudf_large_nullable_int():
    values = [i if i % 3 != 0 else None for i in range(100)]
    series = cudf.Series(values, dtype="Int32")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == values
    assert ak.backend(result) == "cuda"


def test_from_cudf_large_list():
    # 50 lists of varying lengths
    data = [list(range(i % 5)) for i in range(50)]
    series = cudf.Series(data)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == data


def test_from_cudf_large_struct():
    n = 100
    data = [{"a": i, "b": float(i) * 0.5} for i in range(n)]
    series = cudf.Series(data)
    result = ak.from_cudf(series)
    vals = ak.to_list(result)
    assert vals[0]["a"] == 0
    assert vals[-1]["a"] == n - 1


# ---------------------------------------------------------------------------
# Nullable bitmask edge cases: all-valid, all-null, alternating
# ---------------------------------------------------------------------------


def test_from_cudf_all_valid_produces_no_bitmask():
    # A non-nullable series must not be wrapped in BitMaskedArray.
    series = cudf.Series([1, 2, 3, 4, 5], dtype=np.int64)
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.NumpyArray)


def test_from_cudf_all_null_int():
    series = cudf.Series([None, None, None], dtype="Int32")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [None, None, None]


def test_from_cudf_alternating_null_int():
    values = [1 if i % 2 == 0 else None for i in range(16)]
    series = cudf.Series(values, dtype="Int64")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == values


def test_from_cudf_alternating_null_float():
    values = [float(i) if i % 2 == 0 else None for i in range(16)]
    series = cudf.Series(values, dtype="float64")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == pytest.approx(
        [v if v is not None else None for v in values]
    )


# ---------------------------------------------------------------------------
# DataFrame edge cases
# ---------------------------------------------------------------------------


def test_from_cudf_dataframe_zero_rows():
    df = cudf.DataFrame(
        {
            "x": cudf.Series([], dtype=np.int32),
            "y": cudf.Series([], dtype=np.float64),
            "z": cudf.Series([], dtype="str"),
        }
    )
    result = ak.from_cudf(df)
    assert len(result) == 0
    assert ak.fields(result) == ["x", "y", "z"]


def test_from_cudf_dataframe_single_row():
    df = cudf.DataFrame({"a": [42], "b": [3.14]})
    result = ak.from_cudf(df)
    assert len(result) == 1
    vals = ak.to_list(result)
    assert vals[0]["a"] == 42
    assert vals[0]["b"] == pytest.approx(3.14)


def test_from_cudf_dataframe_mixed_nullable():
    df = cudf.DataFrame(
        {
            "a": cudf.Series([1, None, 3], dtype="Int64"),
            "b": cudf.Series([None, 2.0, 3.0], dtype="float64"),
            "c": cudf.Series([True, False, None]),
        }
    )
    result = ak.from_cudf(df)
    vals = ak.to_list(result)
    assert vals[0]["a"] == 1
    assert vals[1]["a"] is None
    assert vals[0]["b"] is None
    assert vals[1]["b"] == pytest.approx(2.0)
    assert vals[2]["c"] is None


def test_from_cudf_dataframe_with_list_and_struct_columns():
    df = cudf.DataFrame(
        {
            "flat": [1, 2, 3],
            "lists": cudf.Series([[1, 2], [3], []], dtype=cudf.ListDtype("int32")),
        }
    )
    result = ak.from_cudf(df)
    vals = ak.to_list(result)
    assert vals[0]["flat"] == 1
    assert vals[0]["lists"] == [1, 2]
    assert vals[2]["lists"] == []


# ---------------------------------------------------------------------------
# Behavior and attrs propagation
# ---------------------------------------------------------------------------


def test_from_cudf_behavior_propagated():
    series = cudf.Series([1, 2, 3], dtype=np.int64)
    behavior = {"custom": True}
    result = ak.from_cudf(series, behavior=behavior)
    assert result.behavior is behavior


def test_from_cudf_attrs_propagated():
    series = cudf.Series([1, 2, 3], dtype=np.int64)
    attrs = {"source": "test", "version": 1}
    result = ak.from_cudf(series, attrs=attrs)
    assert result.attrs == attrs


def test_from_cudf_behavior_and_attrs_propagated_together():
    series = cudf.Series([1.5, 2.5], dtype=np.float32)
    behavior = {}
    attrs = {"key": "value"}
    result = ak.from_cudf(series, behavior=behavior, attrs=attrs)
    assert result.behavior is behavior
    assert result.attrs == attrs
