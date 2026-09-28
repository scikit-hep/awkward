# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import math

import numpy as np
import pytest

import awkward as ak

cudf = pytest.importorskip("cudf")
cp = pytest.importorskip("cupy")
pytest.importorskip("pylibcudf")


def _to_pylibcudf_column(series):
    col = series.to_pylibcudf()
    return col[0] if isinstance(col, tuple) else col


def _data_buffer(plc_column):
    """pylibcudf spells the data buffer `data_buffer` or `data`, as a property
    or a method, depending on the release."""
    for name in ("data_buffer", "data"):
        value = getattr(plc_column, name, None)
        if value is not None:
            return value() if callable(value) else value
    raise AssertionError("pylibcudf column exposes no data buffer")


class TestPrimitives:
    @pytest.mark.parametrize(
        ("dtype", "values"),
        [
            ("int8", [-1, 0, 1]),
            ("int16", [-1, 0, 1]),
            ("int32", [-1, 0, 1]),
            ("int64", [-1, 0, 1]),
            ("uint8", [0, 1, 2]),
            ("uint16", [0, 1, 2]),
            ("uint32", [0, 1, 2]),
            ("uint64", [0, 1, 2]),
            ("float32", [1.5, 2.5, 3.5]),
            ("float64", [1.5, 2.5, 3.5]),
            ("bool", [True, False, True]),
        ],
    )
    def test_numeric_and_bool(self, dtype, values):
        result = ak.from_cudf(cudf.Series(values, dtype=dtype))
        assert ak.to_list(result) == values
        assert ak.backend(result) == "cuda"

    def test_empty_series(self):
        result = ak.from_cudf(cudf.Series([], dtype="int64"))
        assert ak.to_list(result) == []
        assert result.layout.data.dtype == cp.dtype("int64")

    def test_float_nan_passthrough(self):
        # cuDF's default nan_as_null=True would turn the NaN into a null, so
        # it has to be switched off to get a NaN payload into the column.
        result = ak.from_cudf(
            cudf.Series([1.0, math.nan, 3.0], dtype="float64", nan_as_null=False)
        )
        values = ak.to_list(result)
        assert values[0] == 1.0
        assert math.isnan(values[1])
        assert values[2] == 3.0
        assert not result.layout.is_option

    def test_float_nan_as_null(self):
        # With cuDF's default nan_as_null=True the NaN is a null in the column
        # itself, and from_cudf reports it as a missing value.
        result = ak.from_cudf(cudf.Series([1.0, math.nan, 3.0], dtype="float64"))
        assert ak.to_list(result) == [1.0, None, 3.0]
        assert result.layout.is_option


class TestNullable:
    def test_nullable_int(self):
        result = ak.from_cudf(cudf.Series([1, None, 3], dtype="Int64"))
        assert ak.to_list(result) == [1, None, 3]
        assert result.layout.is_option

    def test_all_null(self):
        result = ak.from_cudf(cudf.Series([None, None], dtype="Int64"))
        assert ak.to_list(result) == [None, None]
        assert result.layout.is_option

    def test_non_nullable_is_not_option(self):
        layout = ak.from_cudf(cudf.Series([1, 2, 3], dtype="int64"), highlevel=False)
        assert not layout.is_option


class TestLists:
    def test_list_of_int(self):
        result = ak.from_cudf(cudf.Series([[1, 2], [3]]))
        assert ak.to_list(result) == [[1, 2], [3]]

    def test_list_with_nulls(self):
        result = ak.from_cudf(cudf.Series([[1, 2], None, [3]]))
        assert ak.to_list(result) == [[1, 2], None, [3]]

    def test_empty_lists(self):
        result = ak.from_cudf(cudf.Series([[], [1], []]))
        assert ak.to_list(result) == [[], [1], []]

    def test_nested_list(self):
        result = ak.from_cudf(cudf.Series([[[1, 2]], [[3], []]]))
        assert ak.to_list(result) == [[[1, 2]], [[3], []]]

    def test_empty_list_column(self):
        dtype = cudf.core.dtypes.ListDtype(np.dtype("int64"))
        result = ak.from_cudf(cudf.Series([], dtype=dtype))
        assert ak.to_list(result) == []


class TestStructs:
    def test_simple_dataframe(self):
        result = ak.from_cudf(cudf.DataFrame({"x": [1, 2], "y": [3, 4]}))
        assert ak.to_list(result) == [{"x": 1, "y": 3}, {"x": 2, "y": 4}]
        assert ak.fields(result) == ["x", "y"]

    def test_struct_series(self):
        result = ak.from_cudf(cudf.Series([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]))
        assert ak.to_list(result) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]

    def test_struct_with_nulls(self):
        result = ak.from_cudf(
            cudf.Series([{"x": 1, "y": 1.1}, None, {"x": 3, "y": 3.3}])
        )
        assert ak.to_list(result) == [{"x": 1, "y": 1.1}, None, {"x": 3, "y": 3.3}]

    def test_struct_containing_list_field(self):
        result = ak.from_cudf(cudf.Series([{"x": [1, 2]}, {"x": [3]}]))
        assert ak.to_list(result) == [{"x": [1, 2]}, {"x": [3]}]


class TestStrings:
    def test_plain_strings(self):
        result = ak.from_cudf(cudf.Series(["hello", "world"]))
        assert ak.to_list(result) == ["hello", "world"]

    def test_strings_with_nulls(self):
        result = ak.from_cudf(cudf.Series(["hello", None, "world"]))
        assert ak.to_list(result) == ["hello", None, "world"]

    def test_empty_strings(self):
        result = ak.from_cudf(cudf.Series(["", "x", ""]))
        assert ak.to_list(result) == ["", "x", ""]

    def test_all_empty_strings(self):
        # A column of only empty strings has no character buffer at all.
        result = ak.from_cudf(cudf.Series(["", "", ""]))
        assert ak.to_list(result) == ["", "", ""]

    def test_empty_strings_with_nulls(self):
        result = ak.from_cudf(cudf.Series(["", None, ""]))
        assert ak.to_list(result) == ["", None, ""]

    def test_unicode(self):
        result = ak.from_cudf(cudf.Series(["alpha", "emoji \U0001f600", "delta"]))
        assert ak.to_list(result) == ["alpha", "emoji \U0001f600", "delta"]


class TestDictionary:
    def test_category_encoded_ints(self):
        result = ak.from_cudf(cudf.Series([1, 2, 1], dtype="category"))
        assert ak.to_list(result) == [1, 2, 1]
        assert result.layout.parameter("__array__") == "categorical"

    def test_category_encoded_strings(self):
        result = ak.from_cudf(cudf.Series(["a", "b", "a"], dtype="category"))
        assert ak.to_list(result) == ["a", "b", "a"]
        assert result.layout.parameter("__array__") == "categorical"

    def test_category_with_nulls(self):
        result = ak.from_cudf(cudf.Series(["a", None, "b"], dtype="category"))
        assert ak.to_list(result) == ["a", None, "b"]
        assert result.layout.is_option
        assert result.layout.parameter("__array__") == "categorical"

    def test_category_is_indexed_and_on_device(self):
        layout = ak.from_cudf(
            cudf.Series(["a", "b", "a"], dtype="category"), highlevel=False
        )
        assert isinstance(layout, ak.contents.IndexedArray)
        assert ak.backend(layout) == "cuda"
        # The content holds the distinct categories, not one entry per row.
        assert ak.to_list(ak.Array(layout.content)) == ["a", "b"]

    def test_category_empty(self):
        result = ak.from_cudf(cudf.Series([], dtype="category"))
        assert ak.to_list(result) == []

    def test_category_sliced(self):
        series = cudf.Series(["a", "b", "a", "c"], dtype="category")
        result = ak.from_cudf(series.iloc[1:3])
        assert ak.to_list(result) == ["b", "a"]

    def test_category_in_dataframe(self):
        result = ak.from_cudf(
            cudf.DataFrame({"x": cudf.Series(["a", "b", "a"], dtype="category")})
        )
        assert ak.to_list(result) == [{"x": "a"}, {"x": "b"}, {"x": "a"}]


class TestDataFrame:
    def test_basic(self):
        result = ak.from_cudf(cudf.DataFrame({"x": [1, 2], "y": [3, 4]}))
        assert ak.to_list(result) == [{"x": 1, "y": 3}, {"x": 2, "y": 4}]

    def test_mixed_types(self):
        result = ak.from_cudf(
            cudf.DataFrame({"x": [1, 2], "y": ["one", "two"], "z": [True, False]})
        )
        assert ak.to_list(result) == [
            {"x": 1, "y": "one", "z": True},
            {"x": 2, "y": "two", "z": False},
        ]

    def test_column_order_preserved(self):
        result = ak.from_cudf(cudf.DataFrame({"z": [1], "x": [2], "a": [3]}))
        assert ak.fields(result) == ["z", "x", "a"]


class TestErrors:
    def test_wrong_input_type(self):
        with pytest.raises(TypeError, match=r"cudf\.Series or cudf\.DataFrame"):
            ak.from_cudf([1, 2, 3])

    def test_highlevel_false(self):
        layout = ak.from_cudf(cudf.Series([1, 2, 3], dtype="int64"), highlevel=False)
        assert isinstance(layout, ak.contents.NumpyArray)

    def test_behavior_forwarded(self):
        behavior = {}
        attrs = {"source": "cudf"}
        result = ak.from_cudf(
            cudf.Series([1, 2, 3], dtype="int64"),
            behavior=behavior,
            attrs=attrs,
        )
        assert result.behavior is behavior
        assert result.attrs == attrs


class TestZeroCopy:
    def test_primitive_data_buffer_pointer(self):
        series = cudf.Series([1, 2, 3], dtype="int64")
        plc_col = _to_pylibcudf_column(series)
        layout = ak.from_cudf(series, highlevel=False)
        assert layout.data.data.ptr == _data_buffer(plc_col).ptr


def test_dataframe_basic():
    """DataFrame with int and float columns converts correctly."""
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    result = ak.from_cudf(df)
    assert result.fields == ["x", "y"]
    assert ak.to_list(result["x"]) == [1, 2, 3]
    assert ak.to_list(result["y"]) == [1.0, 2.0, 3.0]


def test_dataframe_field_order():
    """Column order in the RecordArray matches the DataFrame."""
    df = cudf.DataFrame({"b": [10, 20], "a": [1, 2], "c": [True, False]})
    result = ak.from_cudf(df)
    assert result.fields == ["b", "a", "c"]


def test_dataframe_nullable_column():
    """Nullable columns in a DataFrame round-trip correctly."""
    df = cudf.DataFrame({"v": cudf.Series([1, None, 3], dtype="Int64")})
    result = ak.from_cudf(df)
    assert ak.to_list(result["v"]) == [1, None, 3]


def test_dataframe_mixed_nullable_and_non_nullable():
    """Mix of nullable and non-nullable columns in one DataFrame."""
    df = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, 3]),
            "b": cudf.Series([None, 5, None], dtype="Int64"),
        }
    )
    result = ak.from_cudf(df)
    assert ak.to_list(result["a"]) == [1, 2, 3]
    assert ak.to_list(result["b"]) == [None, 5, None]


def test_nullable_mask_with_no_actual_nulls():
    """A column with a null mask allocated but null_count==0."""
    s = cudf.Series([1, 2, 3])
    # This is a no-op but may allocate a validity mask depending on cuDF version.
    s[0:0] = None
    result = ak.from_cudf(s)
    assert ak.to_list(result) == [1, 2, 3]


def test_deeply_nested_struct():
    """Struct inside a struct round-trips correctly."""
    df = cudf.DataFrame(
        {"outer": cudf.Series([{"inner": {"v": 1}}, {"inner": {"v": 2}}])}
    )
    result = ak.from_cudf(df["outer"])
    assert ak.to_list(result) == [{"inner": {"v": 1}}, {"inner": {"v": 2}}]


def test_list_of_structs():
    """List column whose elements are structs."""
    s = cudf.Series([[{"x": 1}, {"x": 2}], [{"x": 3}]])
    result = ak.from_cudf(s)
    assert ak.to_list(result) == [[{"x": 1}, {"x": 2}], [{"x": 3}]]


def test_timestamp_seconds():
    """Timestamp[s] column preserves values."""
    s = cudf.Series(np.array(["2024-01-01", "2024-06-15"], dtype="datetime64[s]"))
    result = ak.from_cudf(s)
    assert result.type.content.primitive == "datetime64[s]"


def test_timestamp_nanoseconds():
    """Timestamp[ns] column preserves values."""
    s = cudf.Series(np.array(["2024-01-01", "2024-06-15"], dtype="datetime64[ns]"))
    result = ak.from_cudf(s)
    assert result.type.content.primitive == "datetime64[ns]"
