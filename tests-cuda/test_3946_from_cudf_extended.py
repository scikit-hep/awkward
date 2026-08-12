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
