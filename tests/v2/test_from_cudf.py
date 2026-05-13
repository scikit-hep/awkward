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
        result = ak.from_cudf(cudf.Series([1.0, math.nan, 3.0], dtype="float64"))
        values = ak.to_list(result)
        assert values[0] == 1.0
        assert math.isnan(values[1])
        assert values[2] == 3.0


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

    def test_unicode(self):
        result = ak.from_cudf(cudf.Series(["alpha", "emoji \U0001f600", "delta"]))
        assert ak.to_list(result) == ["alpha", "emoji \U0001f600", "delta"]


class TestDictionary:
    def test_category_encoded_ints(self):
        result = ak.from_cudf(cudf.Series([1, 2, 1], dtype="category"))
        assert ak.to_list(result) == [1, 2, 1]

    def test_category_encoded_strings(self):
        result = ak.from_cudf(cudf.Series(["a", "b", "a"], dtype="category"))
        assert ak.to_list(result) == ["a", "b", "a"]


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
        assert layout.data.data.ptr == plc_col.data_buffer().ptr
