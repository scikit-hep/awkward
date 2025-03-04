from __future__ import annotations

import pytest

import awkward as ak

cudf = pytest.importorskip("cudf")
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
