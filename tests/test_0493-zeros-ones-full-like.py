# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import datetime
import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak._v2.Array(
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

    assert ak._v2.operations.full_like(array, 12.3).tolist() == [
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

    assert ak._v2.operations.zeros_like(array).tolist() == [
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

    assert ak._v2.operations.ones_like(array).tolist() == [
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

    array = ak._v2.Array([["one", "two", "three"], [], ["four", "five"]])
    assert ak._v2.operations.full_like(array, "hello").tolist() == [
        ["hello", "hello", "hello"],
        [],
        ["hello", "hello"],
    ]
    assert ak._v2.operations.full_like(array, 1).tolist() == [
        ["1", "1", "1"],
        [],
        ["1", "1"],
    ]
    assert ak._v2.operations.full_like(array, 0).tolist() == [
        ["0", "0", "0"],
        [],
        ["0", "0"],
    ]
    assert ak._v2.operations.ones_like(array).tolist() == [
        ["1", "1", "1"],
        [],
        ["1", "1"],
    ]
    assert ak._v2.operations.zeros_like(array).tolist() == [
        ["", "", ""],
        [],
        ["", ""],
    ]

    array = ak._v2.Array([[b"one", b"two", b"three"], [], [b"four", b"five"]])
    assert ak._v2.operations.full_like(array, b"hello").tolist() == [
        [b"hello", b"hello", b"hello"],
        [],
        [b"hello", b"hello"],
    ]
    assert ak._v2.operations.full_like(array, 1).tolist() == [
        [b"1", b"1", b"1"],
        [],
        [b"1", b"1"],
    ]
    assert ak._v2.operations.full_like(array, 0).tolist() == [
        [b"0", b"0", b"0"],
        [],
        [b"0", b"0"],
    ]
    assert ak._v2.operations.ones_like(array).tolist() == [
        [b"1", b"1", b"1"],
        [],
        [b"1", b"1"],
    ]
    assert ak._v2.operations.zeros_like(array).tolist() == [
        [b"", b"", b""],
        [],
        [b"", b""],
    ]


def test_full_like_types():

    array = ak._v2.highlevel.Array(
        np.array(["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]")
    )

    assert ak._v2.operations.full_like(array, "2020-07-27T10:41:11").tolist() == [
        datetime.datetime(2020, 7, 27, 10, 41, 11),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
    ]

    array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[25s]"
    )

    assert ak._v2.operations.full_like(array, "2021-06-03T10:00").tolist() == [
        datetime.datetime(2021, 6, 3, 10, 0),
        datetime.datetime(2021, 6, 3, 10, 0),
        datetime.datetime(2021, 6, 3, 10, 0),
    ]

    array = ak._v2.contents.NumpyArray(np.array([0, 2, 2, 3], dtype="i4"))

    assert str(ak._v2.operations.full_like(array, 11, dtype="i8").type) == "4 * int64"
    assert (
        str(ak._v2.operations.full_like(array, 11, dtype=np.dtype(np.int64)).type)
        == "4 * int64"
    )
    assert (
        str(ak._v2.operations.full_like(array, 11, dtype=np.int64).type) == "4 * int64"
    )
