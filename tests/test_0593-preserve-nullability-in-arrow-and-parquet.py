# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")


def test_list():
    ak_array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    pa_array = ak.to_arrow(ak_array)
    assert str(pa_array.type) == "list<item: double not null>"
    assert pa_array.to_pylist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    ak_array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak_array.layout.offsets, ak.layout.UnmaskedArray(ak_array.layout.content)
        )
    )
    pa_array = ak.to_arrow(ak_array)
    assert str(pa_array.type) == "list<item: double>"
    assert pa_array.to_pylist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    ak_array = ak.Array([[1.1, 2.2, None], [], [4.4, 5.5]])
    pa_array = ak.to_arrow(ak_array)
    assert str(pa_array.type) == "list<item: double>"
    assert pa_array.to_pylist() == [[1.1, 2.2, None], [], [4.4, 5.5]]


def test_record():
    x_content = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5]).layout
    z_content = ak.Array([1, 2, 3, None, 5]).layout

    ak_array = ak.Array(
        ak.layout.RecordArray(
            [x_content, ak.layout.UnmaskedArray(x_content), z_content,], ["x", "y", "z"]
        )
    )
    pa_array = ak.to_arrow(ak_array)
    assert str(pa_array.type) == "struct<x: double not null, y: double, z: int64>"
    assert pa_array.to_pylist() == [
        {"x": 1.1, "y": 1.1, "z": 1},
        {"x": 2.2, "y": 2.2, "z": 2},
        {"x": 3.3, "y": 3.3, "z": 3},
        {"x": 4.4, "y": 4.4, "z": None},
        {"x": 5.5, "y": 5.5, "z": 5},
    ]


def test_union():
    ak_array = ak.Array([1.1, 2.2, None, [1, 2, 3], "hello"])
    py_array = ak.to_arrow(ak_array)
    assert (
        str(py_array.type)
        == "dense_union<0: double=0, 1: list<item: int64 not null>=1, 2: string=2>"
    )
    assert py_array.to_pylist() == [1.1, 2.2, None, [1, 2, 3], "hello"]

    ak_array = ak.Array(
        ak.layout.UnmaskedArray(ak.Array([1.1, 2.2, [1, 2, 3], "hello"]).layout)
    )
    py_array = ak.to_arrow(ak_array)
    assert (
        str(py_array.type)
        == "dense_union<0: double=0, 1: list<item: int64 not null>=1, 2: string=2>"
    )
    assert py_array.to_pylist() == [1.1, 2.2, [1, 2, 3], "hello"]

    ak_array = ak.Array([1.1, 2.2, [1, 2, 3], "hello"])
    py_array = ak.to_arrow(ak_array)
    assert (
        str(py_array.type)
        == "dense_union<0: double not null=0, 1: list<item: int64 not null> not null=1, 2: string not null=2>"
    )
    assert py_array.to_pylist() == [1.1, 2.2, [1, 2, 3], "hello"]
