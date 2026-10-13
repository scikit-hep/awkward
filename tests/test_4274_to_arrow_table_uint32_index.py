# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("pyarrow")
import pyarrow.feather
import pyarrow.parquet


@pytest.fixture(params=["arrow", "feather", "parquet"])
def roundtrip(request, tmp_path):
    def check(array, expected, extensionarray):
        if request.param == "arrow":
            table = ak.to_arrow_table(array, extensionarray=extensionarray)
            restored = ak.from_arrow(table)
        elif request.param == "feather":
            path = Path(mkdtemp(dir=tmp_path)) / "array.feather"
            ak.to_feather(array, path, extensionarray=extensionarray)
            table = pyarrow.feather.read_table(path)
            restored = ak.from_feather(path)
        else:
            path = Path(mkdtemp(dir=tmp_path)) / "array.parquet"
            ak.to_parquet(array, path, extensionarray=extensionarray)
            table = pyarrow.parquet.read_table(path)
            restored = ak.from_parquet(path)
        assert table.to_pylist() == expected
        assert restored.to_list() == expected

    return check


@pytest.mark.parametrize("extensionarray", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "values", [[10, 20, 30], ["a", "bc", "def"], [[1], [], [2, 3]]]
)
def test_unmasked_indexed_field(values, empty, extensionarray, roundtrip):
    indices = [] if empty else [2, 0, 2, 1]
    content = ak.to_layout(values)
    if empty:
        content = content[:0]
    field = ak.contents.IndexedArray(ak.index.IndexU32(indices), content)
    array = ak.Array(ak.contents.UnmaskedArray(ak.contents.RecordArray([field], ["A"])))
    expected = [{"A": values[i]} for i in indices]
    roundtrip(array, expected, extensionarray)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_issue_boolean_reproducer(extensionarray, roundtrip):
    field = ak.contents.IndexedArray(
        ak.index.IndexU32(np.zeros(3, np.uint32)),
        ak.contents.NumpyArray(np.array([False, False])),
    )
    array = ak.Array(ak.contents.UnmaskedArray(ak.contents.RecordArray([field], ["A"])))
    roundtrip(array, [{"A": False}] * 3, extensionarray)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_missing_field_control(extensionarray, roundtrip):
    field = ak.contents.IndexedOptionArray(
        ak.index.Index64([2, -1, 0, 2]), ak.to_layout([10, 20, 30])
    )
    array = ak.Array(ak.contents.RecordArray([field], ["A"]))
    roundtrip(array, [{"A": 30}, {"A": None}, {"A": 10}, {"A": 30}], extensionarray)


@pytest.mark.parametrize("extensionarray", [False, True])
def test_nested_control(extensionarray, roundtrip):
    field = ak.contents.IndexedArray(
        ak.index.IndexU32([2, 0, 2, 1]), ak.to_layout([10, 20, 30])
    )
    records = ak.contents.UnmaskedArray(ak.contents.RecordArray([field], ["A"]))
    array = ak.Array(ak.contents.RecordArray([records], ["nested"]))
    expected = [{"nested": {"A": x}} for x in [30, 10, 30, 20]]
    roundtrip(array, expected, extensionarray)
