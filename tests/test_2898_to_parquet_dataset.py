from __future__ import annotations

import os

import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")


def simple_test(tmp_path):
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    array1 = ak.Array([[1.1, 2.2, 3.3, 4.4], [4.0], [4.4, 5.5]])
    array2 = ak.Array([[1.0, 3.0, 3.3, 4.4], [4.0], [4.4, 10.0], [11.11]])
    ak.to_parquet(
        array, os.path.join(tmp_path, "arr1.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array1, os.path.join(tmp_path, "arr2.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array2, os.path.join(tmp_path, "arr3.parquet"), parquet_compliant_nested=True
    )

    ak.to_parquet_dataset(tmp_path, filenames="arr[1-3].parquet")
    assert os.path.exists(os.path.join(tmp_path, "_common_metadata"))
    assert os.path.exists(os.path.join(tmp_path, "_metadata"))

    with_metadata = ak.from_parquet(tmp_path)
    print(with_metadata.to_list())
    assert with_metadata.tolist() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4],
        [4.0],
        [4.4, 5.5],
        [1.0, 3.0, 3.3, 4.4],
        [4.0],
        [4.4, 10.0],
        [11.11],
    ]


def complex_test(tmp_path):
    array1 = ak.Array(
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]
    )
    array2 = ak.Array([{"x": 1.8, "y": [3, 5, 6]}])
    array3 = ak.Array([{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}])
    ak.to_parquet(
        array1, os.path.join(tmp_path, "arr1.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array2, os.path.join(tmp_path, "arr2.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array3, os.path.join(tmp_path, "arr3.parquet"), parquet_compliant_nested=True
    )

    ak.to_parquet_dataset(tmp_path)

    assert os.path.exists(os.path.join(tmp_path, "_common_metadata"))
    assert os.path.exists(os.path.join(tmp_path, "_metadata"))

    with_metadata = ak.from_parquet(tmp_path)
    assert with_metadata.tolist() == [
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [1, 2]},
        {"x": 3.3, "y": [1, 2, 3]},
        {"x": 1.8, "y": [3, 5, 6]},
        {"x": 4.4, "y": [1, 2, 3, 4]},
        {"x": 5.5, "y": [1, 2, 3, 4, 5]},
    ]


def test_filenames(tmp_path):
    array = ak.Array(
        [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]
    )
    array1 = ak.Array([{"x": 1.8, "y": [3, 5, 6]}])
    array2 = ak.Array([{"x": 4.4, "y": [1, 2, 3, 4]}, {"x": 5.5, "y": [1, 2, 3, 4, 5]}])
    ak.to_parquet(
        array, os.path.join(tmp_path, "arr1.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array1, os.path.join(tmp_path, "arr2.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array2, os.path.join(tmp_path, "arr3.parquet"), parquet_compliant_nested=True
    )

    ak.to_parquet_dataset(
        tmp_path, filenames=["arr1.parquet", "arr2.parquet", "arr3.parquet"]
    )

    assert os.path.exists(os.path.join(tmp_path, "_common_metadata"))
    assert os.path.exists(os.path.join(tmp_path, "_metadata"))

    with_metadata = ak.from_parquet(tmp_path)
    assert with_metadata.tolist() == [
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [1, 2]},
        {"x": 3.3, "y": [1, 2, 3]},
        {"x": 1.8, "y": [3, 5, 6]},
        {"x": 4.4, "y": [1, 2, 3, 4]},
        {"x": 5.5, "y": [1, 2, 3, 4, 5]},
    ]


def test_wildcard(tmp_path):
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    array1 = ak.Array([[1.1, 2.2, 3.3, 4.4], [4.0], [4.4, 5.5]])
    array2 = ak.Array([[1.0, 3.0, 3.3, 4.4], [4.0], [4.4, 10.0], [11.11]])
    ak.to_parquet(
        array, os.path.join(tmp_path, "arr1.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array1, os.path.join(tmp_path, "arr2.parquet"), parquet_compliant_nested=True
    )
    ak.to_parquet(
        array2, os.path.join(tmp_path, "arr3.parquet"), parquet_compliant_nested=True
    )

    ak.to_parquet_dataset(tmp_path, filenames="arr?.parquet")

    with_metadata = ak.from_parquet(tmp_path)
    assert with_metadata.tolist() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [1.1, 2.2, 3.3, 4.4],
        [4.0],
        [4.4, 5.5],
        [1.0, 3.0, 3.3, 4.4],
        [4.0],
        [4.4, 10.0],
        [11.11],
    ]
