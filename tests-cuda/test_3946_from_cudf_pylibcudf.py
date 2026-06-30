# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

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


def _device_values(series):
    plc_column = series.to_pylibcudf()
    plc_column = plc_column[0] if isinstance(plc_column, tuple) else plc_column
    data = cp.asarray(plc_column.data_buffer(), dtype=series.dtype)
    if plc_column.offset() != 0 or data.shape[0] != plc_column.size():
        data = data[plc_column.offset() : plc_column.offset() + plc_column.size()]
    return data


def test_from_cudf_int64():
    series = cudf.Series([1, 2, 3, 4], dtype=np.int64)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [1, 2, 3, 4]
    assert ak.backend(result) == "cuda"


def test_from_cudf_float32():
    series = cudf.Series([0.5, 1.5, 2.5], dtype=np.float32)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == pytest.approx([0.5, 1.5, 2.5])
    assert result.layout.data.dtype == cp.dtype(np.float32)


def test_from_cudf_preserves_int32_dtype():
    series = cudf.Series([1, 2, 3], dtype=np.int32)
    result = ak.from_cudf(series)
    assert result.layout.data.dtype == cp.dtype(np.int32)


def test_from_cudf_uint8():
    series = cudf.Series([0, 127, 255], dtype=np.uint8)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [0, 127, 255]


def test_from_cudf_float64():
    series = cudf.Series([1.1, 2.2, 3.3], dtype=np.float64)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == pytest.approx([1.1, 2.2, 3.3])


def test_from_cudf_negative_values():
    series = cudf.Series([-3, -1, 0, 1, 3], dtype=np.int64)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [-3, -1, 0, 1, 3]


def test_from_cudf_single_element():
    series = cudf.Series([42], dtype=np.int64)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [42]


def test_from_cudf_empty_series():
    series = cudf.Series([], dtype=np.int64)
    result = ak.from_cudf(series)
    assert ak.to_list(result) == []
    assert result.layout.data.dtype == cp.dtype(np.int64)


def test_from_cudf_boolean():
    series = cudf.Series([True, False, True, False])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [True, False, True, False]
    assert ak.backend(result) == "cuda"


def test_from_cudf_nullable_int():
    series = cudf.Series([1, None, 3], dtype="Int64")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [1, None, 3]
    assert ak.backend(result) == "cuda"


def test_from_cudf_nullable_float():
    series = cudf.Series([1.1, None, 3.3], dtype="float64")
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [1.1, None, 3.3]


def test_from_cudf_list_of_int():
    series = cudf.Series([[1, 2, 3], [], [4, 5]])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[1, 2, 3], [], [4, 5]]
    assert ak.backend(result) == "cuda"


def test_from_cudf_list_nested():
    series = cudf.Series([[[1, 2], [3]], [[4]]])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [[[1, 2], [3]], [[4]]]


def test_from_cudf_struct():
    series = cudf.Series([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]
    assert ak.backend(result) == "cuda"


def test_from_cudf_struct_with_null():
    series = cudf.Series([{"x": 1, "y": 1.1}, None, {"x": 3, "y": 3.3}])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == [{"x": 1, "y": 1.1}, None, {"x": 3, "y": 3.3}]


def test_from_cudf_strings():
    series = cudf.Series(["hello", "world", "awkward"])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == ["hello", "world", "awkward"]
    assert ak.backend(result) == "cuda"


def test_from_cudf_strings_with_null():
    series = cudf.Series(["hello", None, "awkward"])
    result = ak.from_cudf(series)
    assert ak.to_list(result) == ["hello", None, "awkward"]


def test_from_cudf_respects_arrow_offset():
    series = cudf.Series([10, 20, 30, 40], dtype=np.int64)
    sliced = series[1:3]
    result = ak.from_cudf(sliced)
    assert ak.to_list(result) == [20, 30]


def test_from_cudf_zero_copy():
    series = cudf.Series([10, 20, 30, 40], dtype=np.int32)
    result = ak.from_cudf(series)
    assert cp.shares_memory(result.layout.data, _device_values(series))


def test_from_cudf_highlevel_false():
    series = cudf.Series([1, 2, 3], dtype=np.int64)
    layout = ak.from_cudf(series, highlevel=False)
    assert isinstance(layout, ak.contents.NumpyArray)
    assert ak.backend(layout) == "cuda"


def test_from_cudf_preserves_behavior_and_attrs():
    series = cudf.Series([1, 2, 3], dtype=np.int64)
    behavior = {}
    attrs = {"source": "cudf"}
    result = ak.from_cudf(series, behavior=behavior, attrs=attrs)
    assert result.behavior is behavior
    assert result.attrs == attrs


def test_from_cudf_rejects_wrong_input():
    with pytest.raises(TypeError, match=r"cudf\.Series"):
        ak.from_cudf(np.array([1, 2, 3]))


def test_from_cudf_rejects_wrong_input_list():
    with pytest.raises(TypeError, match=r"cudf\.Series"):
        ak.from_cudf([1, 2, 3])
