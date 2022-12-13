# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import datetime

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}],
            [],
            [
                {"x": 3.3, "y": [1, 2, None, 3]},
                False,
                False,
                True,
                {"x": 4.4, "y": [1, 2, None, 3, 4]},
            ],
        ]
    )

    assert ak.operations.full_like(array, 12.3).to_list() == [
        [{"x": 12.3, "y": []}, {"x": 12.3, "y": [12]}, {"x": 12.3, "y": [12, 12]}],
        [],
        [
            {"x": 12.3, "y": [12, 12, None, 12]},
            True,
            True,
            True,
            {"x": 12.3, "y": [12, 12, None, 12, 12]},
        ],
    ]

    assert ak.operations.zeros_like(array).to_list() == [
        [{"x": 0.0, "y": []}, {"x": 0.0, "y": [0]}, {"x": 0.0, "y": [0, 0]}],
        [],
        [
            {"x": 0.0, "y": [0, 0, None, 0]},
            False,
            False,
            False,
            {"x": 0.0, "y": [0, 0, None, 0, 0]},
        ],
    ]

    assert ak.operations.ones_like(array).to_list() == [
        [{"x": 1.0, "y": []}, {"x": 1.0, "y": [1]}, {"x": 1.0, "y": [1, 1]}],
        [],
        [
            {"x": 1.0, "y": [1, 1, None, 1]},
            True,
            True,
            True,
            {"x": 1.0, "y": [1, 1, None, 1, 1]},
        ],
    ]

    array = ak.Array([["one", "two", "three"], [], ["four", "five"]])
    assert ak.operations.full_like(array, "hello").to_list() == [
        ["hello", "hello", "hello"],
        [],
        ["hello", "hello"],
    ]
    assert ak.operations.full_like(array, 1).to_list() == [
        ["1", "1", "1"],
        [],
        ["1", "1"],
    ]
    assert ak.operations.full_like(array, 0).to_list() == [
        ["0", "0", "0"],
        [],
        ["0", "0"],
    ]
    assert ak.operations.ones_like(array).to_list() == [
        ["1", "1", "1"],
        [],
        ["1", "1"],
    ]
    assert ak.operations.zeros_like(array).to_list() == [
        ["", "", ""],
        [],
        ["", ""],
    ]

    array = ak.Array([[b"one", b"two", b"three"], [], [b"four", b"five"]])
    assert ak.operations.full_like(array, b"hello").to_list() == [
        [b"hello", b"hello", b"hello"],
        [],
        [b"hello", b"hello"],
    ]
    assert ak.operations.full_like(array, 1).to_list() == [
        [b"1", b"1", b"1"],
        [],
        [b"1", b"1"],
    ]
    assert ak.operations.full_like(array, 0).to_list() == [
        [b"0", b"0", b"0"],
        [],
        [b"0", b"0"],
    ]
    assert ak.operations.ones_like(array).to_list() == [
        [b"1", b"1", b"1"],
        [],
        [b"1", b"1"],
    ]
    assert ak.operations.zeros_like(array).to_list() == [
        [b"", b"", b""],
        [],
        [b"", b""],
    ]


def test_full_like_types():

    array = ak.highlevel.Array(
        np.array(["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]")
    )

    assert ak.operations.full_like(array, "2020-07-27T10:41:11").to_list() == [
        datetime.datetime(2020, 7, 27, 10, 41, 11),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
    ]

    array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[25s]"
    )

    assert ak.operations.full_like(array, "2021-06-03T10:00").to_list() == [
        datetime.datetime(2021, 6, 3, 10, 0),
        datetime.datetime(2021, 6, 3, 10, 0),
        datetime.datetime(2021, 6, 3, 10, 0),
    ]

    array = ak.contents.NumpyArray(np.array([0, 2, 2, 3], dtype="i4"))

    assert str(ak.operations.full_like(array, 11, dtype="i8").type) == "4 * int64"
    assert (
        str(ak.operations.full_like(array, 11, dtype=np.dtype(np.int64)).type)
        == "4 * int64"
    )
    assert str(ak.operations.full_like(array, 11, dtype=np.int64).type) == "4 * int64"
