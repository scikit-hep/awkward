from __future__ import annotations

import numpy as np
import pytest
from packaging.version import parse as parse_version

import awkward as ak

cudf = pytest.importorskip("cudf", exc_type=ImportError)
cupy = pytest.importorskip("cupy")


def test_jagged():
    arr = ak.Array([[[1, 2, 3], [], [3, 4]], []])
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [[[1, 2, 3], [], [3, 4]], []]


def test_nested():
    arr = ak.Array(
        [{"a": 0, "b": 1.0, "c": {"d": 0}}, {"a": 1, "b": 0.0, "c": {"d": 1}}]
    )
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [
        {"a": 0, "b": 1.0, "c": {"d": 0}},
        {"a": 1, "b": 0.0, "c": {"d": 1}},
    ]


def test_null():
    arr = ak.Array([12, None, 21, 12])
    # calls ByteMaskedArray._to_cudf not NumpyArray
    out = ak.to_cudf(arr)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [12, None, 21, 12]

    # True is valid, LSB order
    arr2 = ak.Array(arr.layout.to_BitMaskedArray(True, True))
    out = ak.to_cudf(arr2)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [12, None, 21, 12]

    # reversed LSB (should be rare, involves extra work!)
    arr3 = ak.Array(arr.layout.to_BitMaskedArray(True, False))
    out = ak.to_cudf(arr3)
    assert isinstance(out, cudf.Series)
    assert out.to_arrow().tolist() == [12, None, 21, 12]


def test_strings():
    arr = ak.Array(["hey", "hi", "hum"])
    out = ak.to_cudf(arr)
    assert out.to_arrow().tolist() == ["hey", "hi", "hum"]

    arr = ak.Array(["hey", "hi", None, "hum"])
    out = ak.to_cudf(arr)
    assert out.to_arrow().tolist() == ["hey", "hi", None, "hum"]


@pytest.mark.xfail(
    parse_version(cudf.__version__) >= parse_version("25.12.00"),
    reason="cudf internals changed since v25.12.00",
)
def test_indexed():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.array([2, 0, 2, 1], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([10, 20, 30], dtype=np.int64)),
    )
    out = ak.to_cudf(ak.Array(layout))
    assert out.to_arrow().tolist() == [30, 10, 30, 20]


@pytest.mark.xfail(
    parse_version(cudf.__version__) >= parse_version("25.12.00"),
    reason="cudf internals changed since v25.12.00",
)
def test_unmasked():
    layout = ak.contents.UnmaskedArray(
        ak.contents.NumpyArray(np.array([1.5, 2.5, 3.5], dtype=np.float64))
    )
    out = ak.to_cudf(ak.Array(layout))
    assert out.to_arrow().tolist() == [1.5, 2.5, 3.5]


@pytest.mark.xfail(
    parse_version(cudf.__version__) >= parse_version("25.12.00"),
    reason="cudf internals changed since v25.12.00",
)
def test_emptyarray():
    out = ak.to_cudf(ak.Array(ak.contents.EmptyArray()))
    assert out.to_arrow().tolist() == []


@pytest.mark.xfail(
    parse_version(cudf.__version__) >= parse_version("25.12.00"),
    reason="cudf internals changed since v25.12.00",
)
def test_indexedoption():
    layout = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 2, 1], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([10, 20, 30], dtype=np.int64)),
    )
    out = ak.to_cudf(ak.Array(layout))
    assert out.to_arrow().tolist() == [10, None, 30, 20]


@pytest.mark.xfail(
    parse_version(cudf.__version__) >= parse_version("25.12.00"),
    reason="cudf internals changed since v25.12.00",
)
def test_listarray():
    layout = ak.contents.ListArray(
        ak.index.Index64(np.array([0, 3, 3], dtype=np.int64)),
        ak.index.Index64(np.array([3, 3, 5], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5], dtype=np.int64)),
    )
    out = ak.to_cudf(ak.Array(layout))
    assert out.to_arrow().tolist() == [[1, 2, 3], [], [4, 5]]
