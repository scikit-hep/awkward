# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import os

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pyarrow = pytest.importorskip("pyarrow")


def test_list_to_arrow():
    ak_array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    pa_array = ak.to_arrow(ak_array)
    assert str(pa_array.type) == "large_list<item: double not null>"
    assert pa_array.to_pylist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    ak_array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak_array.layout.offsets, ak.layout.UnmaskedArray(ak_array.layout.content)
        )
    )
    pa_array = ak.to_arrow(ak_array)
    assert str(pa_array.type) == "large_list<item: double>"
    assert pa_array.to_pylist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    ak_array = ak.Array([[1.1, 2.2, None], [], [4.4, 5.5]])
    pa_array = ak.to_arrow(ak_array)
    assert str(pa_array.type) == "large_list<item: double>"
    assert pa_array.to_pylist() == [[1.1, 2.2, None], [], [4.4, 5.5]]


def test_record_to_arrow():
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


def test_union_to_arrow():
    ak_array = ak.Array([1.1, 2.2, None, [1, 2, 3], "hello"])
    pa_array = ak.to_arrow(ak_array)
    assert (
        str(pa_array.type)
        == "dense_union<0: double=0, 1: large_list<item: int64 not null>=1, 2: string=2>"
    )
    assert pa_array.to_pylist() == [1.1, 2.2, None, [1, 2, 3], "hello"]

    ak_array = ak.Array(
        ak.layout.UnmaskedArray(ak.Array([1.1, 2.2, [1, 2, 3], "hello"]).layout)
    )
    pa_array = ak.to_arrow(ak_array)
    assert (
        str(pa_array.type)
        == "dense_union<0: double=0, 1: large_list<item: int64 not null>=1, 2: string=2>"
    )
    assert pa_array.to_pylist() == [1.1, 2.2, [1, 2, 3], "hello"]

    ak_array = ak.Array([1.1, 2.2, [1, 2, 3], "hello"])
    pa_array = ak.to_arrow(ak_array)
    assert (
        str(pa_array.type)
        == "dense_union<0: double not null=0, 1: large_list<item: int64 not null> not null=1, 2: string not null=2>"
    )
    assert pa_array.to_pylist() == [1.1, 2.2, [1, 2, 3], "hello"]


def test_list_from_arrow():
    original = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert str(reconstituted.type) == "3 * var * float64"
    assert reconstituted.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    original = ak.Array([[1.1, 2.2, None], [], [4.4, 5.5]])
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert str(reconstituted.type) == "3 * var * ?float64"
    assert reconstituted.tolist() == [[1.1, 2.2, None], [], [4.4, 5.5]]

    original = ak.Array([[1.1, 2.2, 3.3], [], None, [4.4, 5.5]])
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert str(reconstituted.type) == "4 * option[var * float64]"
    assert reconstituted.tolist() == [[1.1, 2.2, 3.3], [], None, [4.4, 5.5]]

    original = ak.Array([[1.1, 2.2, None], [], None, [4.4, 5.5]])
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert str(reconstituted.type) == "4 * option[var * ?float64]"
    assert reconstituted.tolist() == [[1.1, 2.2, None], [], None, [4.4, 5.5]]


def test_record_from_arrow():
    x_content = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5]).layout
    z_content = ak.Array([1, 2, 3, None, 5]).layout

    original = ak.Array(
        ak.layout.RecordArray(
            [x_content, ak.layout.UnmaskedArray(x_content), z_content,], ["x", "y", "z"]
        )
    )
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert str(reconstituted.type) == '5 * {"x": float64, "y": ?float64, "z": ?int64}'
    assert reconstituted.tolist() == [
        {"x": 1.1, "y": 1.1, "z": 1},
        {"x": 2.2, "y": 2.2, "z": 2},
        {"x": 3.3, "y": 3.3, "z": 3},
        {"x": 4.4, "y": 4.4, "z": None},
        {"x": 5.5, "y": 5.5, "z": 5},
    ]

    original = ak.Array(
        ak.layout.ByteMaskedArray(
            ak.layout.Index8(np.array([False, True, False, False, False], np.int8)),
            original.layout,
            valid_when=False,
        )
    )
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert str(reconstituted.type) == '5 * ?{"x": float64, "y": ?float64, "z": ?int64}'
    assert reconstituted.tolist() == [
        {"x": 1.1, "y": 1.1, "z": 1},
        None,
        {"x": 3.3, "y": 3.3, "z": 3},
        {"x": 4.4, "y": 4.4, "z": None},
        {"x": 5.5, "y": 5.5, "z": 5},
    ]


def test_union_from_arrow():
    original = ak.Array([1.1, 2.2, [1, 2, 3], "hello"])
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert str(reconstituted.type) == "4 * union[float64, var * int64, string]"
    assert reconstituted.tolist() == [1.1, 2.2, [1, 2, 3], "hello"]

    original = ak.Array([1.1, 2.2, None, [1, 2, 3], "hello"])
    pa_array = ak.to_arrow(original)
    reconstituted = ak.from_arrow(pa_array)
    assert (
        str(reconstituted.type)
        == "5 * union[?float64, option[var * int64], option[string]]"
    )
    assert reconstituted.tolist() == [1.1, 2.2, None, [1, 2, 3], "hello"]


def test_to_arrow_table():
    assert ak.from_arrow(
        ak.to_arrow_table(
            ak.Array([[{"x": 1.1, "y": [1]}], [], [{"x": 2.2, "y": [1, 2]}]]),
            explode_records=True,
        )
    ).tolist() == [
        {"x": [1.1], "y": [[1]]},
        {"x": [], "y": []},
        {"x": [2.2], "y": [[1, 2]]},
    ]
    assert ak.from_arrow(
        ak.to_arrow_table(ak.Array([{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}]))
    ).tolist() == [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}]


def test_to_parquet(tmp_path):
    original = ak.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
            [],
            [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
            [],
            [],
            [
                {"x": 6, "y": 6.6},
                {"x": 7, "y": 7.7},
                {"x": 8, "y": 8.8},
                {"x": 9, "y": 9.9},
            ],
        ]
    )

    ak.to_parquet(original, os.path.join(tmp_path, "data.parquet"))
    reconstituted = ak.from_parquet(os.path.join(tmp_path, "data.parquet"))
    assert reconstituted.tolist() == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        [],
        [],
        [
            {"x": 6, "y": 6.6},
            {"x": 7, "y": 7.7},
            {"x": 8, "y": 8.8},
            {"x": 9, "y": 9.9},
        ],
    ]
    assert str(reconstituted.type) == '6 * var * {"x": int64, "y": float64}'


def test_to_parquet_2(tmp_path):
    array = ak.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": None}],
            [],
            [{"x": 3.3, "y": [1, 2, 3]}, None, {"x": 4.4, "y": [1, 2, 3, 4]}],
        ]
    )
    assert str(array.type) == '3 * var * ?{"x": float64, "y": option[var * int64]}'
    ak.to_parquet(array, os.path.join(tmp_path, "complicated-example.parquet"))
    array2 = ak.from_parquet(os.path.join(tmp_path, "complicated-example.parquet"))
    assert str(array2.type) == str(array.type)
    assert array2.tolist() == array.tolist()


def test_to_table_2():
    array = ak.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": None}],
            [],
            [{"x": 3.3, "y": [1, 2, 3]}, None, {"x": 4.4, "y": [1, 2, 3, 4]}],
        ]
    )
    assert str(array.type) == '3 * var * ?{"x": float64, "y": option[var * int64]}'
    pa_table = ak.to_arrow_table(array)
    array2 = ak.from_arrow(pa_table)
    assert str(array2.type) == str(array.type)
    assert array2.tolist() == array.tolist()
