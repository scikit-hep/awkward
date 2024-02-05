from __future__ import annotations

import os

import pytest

import awkward as ak

pytest.importorskip("pyarrow.parquet")


def generator(n):
    arr = [
        ak.Array(
            [
                {"x": 1.1, "y": [1], "z": [9, 0, 10]},
                {"x": 2.2, "y": [1, 2], "z": [9, 11, 14]},
                {"x": 3.3, "y": [1, 2, 3], "z": [12, 14]},
            ]
        ),
        ak.Array(
            [
                {"x": 1.1, "y": [1], "z": [8]},
                {"x": 2.2, "y": [1, 2], "z": [6, 2, 10]},
                {"x": 3.3, "y": [1, 2, 3], "z": [9, 14]},
            ]
        ),
        ak.Array(
            [
                {"x": 1.1, "y": [1], "z": [9, 11, 14]},
                {"x": 4.4, "y": [1, 2, 3, 4], "z": [1, 14]},
                {"x": 5.5, "y": [1, 2, 3, 4, 5], "z": [9]},
            ]
        ),
        ak.Array(
            [
                {"x": 3.3, "y": [1, 4], "z": [9, 11, 14]},
                {"x": 4.4, "y": [1], "z": [1]},
                {"x": 5.5, "y": [1, 4, 5], "z": [9, 10]},
            ]
        ),
        ak.Array(
            [
                {"x": 66.0, "y": [1, 2, 4], "z": [9, 11, 14]},
                {"x": 4.4, "y": [1, 2], "z": [1, 14]},
                {"x": 5.5, "y": [1, 2, 3, 4, 5], "z": [9, 2]},
            ]
        ),
    ]

    for _ in range(n):
        yield arr[_]


def test_simple(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")

    arr = [
        ak.Array(
            [
                {"x": 1.1, "y": [1]},
                {"x": 2.2, "y": [1, 2]},
                {"x": 3.3, "y": [1, 2, 3]},
                {"x": 1.8, "y": [3, 5, 6]},
                {"x": 4.4, "y": [1, 2, 3, 4]},
                {"x": 5.5, "y": [5, 3, 2, 6]},
            ]
        ),
        ak.Array(
            [
                {"x": 1.1, "y": [1]},
                {"x": 2.2, "y": [1, 2]},
                {"x": 3.3, "y": [1, 2, 3]},
                {"x": 4.4, "y": [1, 2, 3, 4]},
                {"x": 1.8, "y": [3, 5, 6]},
                {"x": 5.5, "y": [1, 2, 3, 4, 5]},
            ]
        ),
        ak.Array(
            [
                {"x": 1.1, "y": [1]},
                {"x": 4.4, "y": [1, 2, 3, 4]},
                {"x": 5.5, "y": [1, 2, 3, 4, 5]},
                {"x": 2.2, "y": [1, 2]},
                {"x": 3.3, "y": [1, 2, 3]},
                {"x": 1.8, "y": [3, 5, 6]},
            ]
        ),
    ]

    check = ak.Array(
        [
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [1, 2]},
            {"x": 3.3, "y": [1, 2, 3]},
            {"x": 1.8, "y": [3, 5, 6]},
            {"x": 4.4, "y": [1, 2, 3, 4]},
            {"x": 5.5, "y": [5, 3, 2, 6]},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [1, 2]},
            {"x": 3.3, "y": [1, 2, 3]},
            {"x": 4.4, "y": [1, 2, 3, 4]},
            {"x": 1.8, "y": [3, 5, 6]},
            {"x": 5.5, "y": [1, 2, 3, 4, 5]},
            {"x": 1.1, "y": [1]},
            {"x": 4.4, "y": [1, 2, 3, 4]},
            {"x": 5.5, "y": [1, 2, 3, 4, 5]},
            {"x": 2.2, "y": [1, 2]},
            {"x": 3.3, "y": [1, 2, 3]},
            {"x": 1.8, "y": [3, 5, 6]},
        ]
    )

    iterator = (i for i in arr)

    metadata = ak.to_parquet_row_groups(iterator, filename)
    assert len(arr) == metadata.num_row_groups

    data = ak.from_parquet(filename)

    assert data.tolist() == check.tolist()

    with pytest.raises(TypeError):
        ak.to_parquet_row_groups(check, filename)

    with pytest.raises(ValueError):
        ak.to_parquet_row_groups([], filename)


def test_general(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    iterator = generator(3)

    metadata = ak.to_parquet_row_groups(iterator, filename)
    assert metadata.num_row_groups == 3

    data = ak.from_parquet(filename)
    assert len(data) == 9

    assert data["x"].tolist() == [1.1, 2.2, 3.3, 1.1, 2.2, 3.3, 1.1, 4.4, 5.5]
    assert data["y"].tolist() == [
        [1],
        [1, 2],
        [1, 2, 3],
        [1],
        [1, 2],
        [1, 2, 3],
        [1],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    ]
    assert data["z"].tolist() == [
        [9, 0, 10],
        [9, 11, 14],
        [12, 14],
        [8],
        [6, 2, 10],
        [9, 14],
        [9, 11, 14],
        [1, 14],
        [9],
    ]


def test_params(tmp_path):
    filename = os.path.join(tmp_path, "whatever.parquet")
    iterator = generator(5)

    metadata = ak.to_parquet_row_groups(
        iterator,
        filename,
        extensionarray=False,
        count_nulls=False,
        row_group_size=64 * 64 * 64,
    )
    assert metadata.num_row_groups == 5

    data = ak.from_parquet(filename)
    assert len(data) == 15

    assert data["x"].tolist() == [
        1.1,
        2.2,
        3.3,
        1.1,
        2.2,
        3.3,
        1.1,
        4.4,
        5.5,
        3.3,
        4.4,
        5.5,
        66.0,
        4.4,
        5.5,
    ]
    assert data["y"].tolist() == [
        [1],
        [1, 2],
        [1, 2, 3],
        [1],
        [1, 2],
        [1, 2, 3],
        [1],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 4],
        [1],
        [1, 4, 5],
        [1, 2, 4],
        [1, 2],
        [1, 2, 3, 4, 5],
    ]
    assert data["z"].tolist() == [
        [9, 0, 10],
        [9, 11, 14],
        [12, 14],
        [8],
        [6, 2, 10],
        [9, 14],
        [9, 11, 14],
        [1, 14],
        [9],
        [9, 11, 14],
        [1],
        [9, 10],
        [9, 11, 14],
        [1, 14],
        [9, 2],
    ]
