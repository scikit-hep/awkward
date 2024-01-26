from __future__ import annotations

import os

import awkward as ak


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
    try:
        os.remove(os.path.join(tmp_path, "_common_metadata"))
        os.remove(os.path.join(tmp_path, "_metadata"))
    except:
        print("not there")

    no_metadata = ak.from_parquet(tmp_path)
    assert no_metadata.tolist() != [
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]],
        [[1.1, 2.2, 3.3, 4.4], [4.0], [4.4, 5.5]],
        [[1.0, 3.0, 3.3, 4.4], [4.0], [4.4, 10.0], [11.11]],
    ]

    ak.to_parquet_dataset(tmp_path, ["arr1.parquet", "arr2.parquet", "arr3.parquet"])

    assert os.path.exists(os.path.join(tmp_path, "_common_metadata"))
    assert os.path.exists(os.path.join(tmp_path, "_metadata"))

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


def complex_test(tmp_path):
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
    try:
        os.remove(os.path.join(tmp_path, "_common_metadata"))
        os.remove(os.path.join(tmp_path, "_metadata"))
    except:
        print("not there")

    no_metadata = ak.from_parquet(tmp_path)
    assert no_metadata.tolist() != [
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [1, 2]},
        {"x": 3.3, "y": [1, 2, 3]},
        {"x": 1.8, "y": [3, 5, 6]},
        {"x": 4.4, "y": [1, 2, 3, 4]},
        {"x": 5.5, "y": [1, 2, 3, 4, 5]},
    ]

    ak.to_parquet_dataset(tmp_path, ["arr1.parquet", "arr2.parquet", "arr3.parquet"])

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
