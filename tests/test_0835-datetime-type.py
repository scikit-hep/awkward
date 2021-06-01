# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_date_time():

    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )

    array = ak.Array(numpy_array)
    assert str(array.type) == "3 * datetime64"
    assert array.tolist() == [
        np.datetime64("2020-07-27T10:41:11"),
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
    ]
    for i in range(len(array)):
        assert ak.to_numpy(array)[i] == numpy_array[i]

    date_time = np.datetime64("2020-07-27T10:41:11.200000011", "us")
    array1 = ak.Array(np.array(["2020-07-27T10:41:11.200000011"], "datetime64[us]"))
    assert np.datetime64(array1[0], "us") == date_time

    # FIXME: this prints '2020-07-27T10:41:11.200000'
    print(ak.to_numpy(array1))

    assert ak.to_list(ak.from_iter(array1)) == [
        np.datetime64("2020-07-27T10:41:11.200000")
    ]

    assert ak.max(array) == numpy_array[0]
    assert ak.min(array) == numpy_array[1]


def test_datetime64_ArrayBuilder():
    builder = ak.layout.ArrayBuilder()
    dt = np.datetime64("2020-03-27T10:41:12", "25us")
    dt1 = np.datetime64("2020-03-27T10:41", "15s")
    dt2 = np.datetime64("2020-05")
    # FIXME: do we need to support this?
    # builder.datetime64(dt.astype(np.int64), "datetime64[s]")
    builder.datetime64(dt1)
    builder.datetime64("2020-03-27T10:41:11")
    builder.datetime64(dt)
    builder.datetime64("2021-03-27")
    builder.datetime64("2020-03-27T10:41:13")
    builder.datetime64(dt2)
    builder.datetime64("2020-05-01T00:00:00.000000")
    builder.datetime64("2020-07-27T10:41:11.200000")

    print(builder.snapshot())
    assert ak.to_list(builder.snapshot()) == [
        np.datetime64("2020-03-27T10:41:00.000000"),
        np.datetime64("2020-03-27T10:41:11.000000"),
        np.datetime64("2020-03-27T10:41:12.000000"),
        np.datetime64("2021-03-27T00:00:00.000000"),
        np.datetime64("2020-03-27T10:41:13.000000"),
        np.datetime64("2020-05-01T20:56:24.000000"),  # FIXME? prescission
        np.datetime64("2020-05-01T00:00:00.000000"),
        np.datetime64("2020-07-27T10:41:11.200000"),
    ]


def test_count():
    array = ak.Array(
        [
            [
                [np.datetime64("2022"), np.datetime64("2023"), np.datetime64("2025")],
                [],
                [np.datetime64("2027"), np.datetime64("2011")],
                [np.datetime64("2013")],
            ],
            [],
            [[np.datetime64("2017"), np.datetime64("2019")], [np.datetime64("2023")]],
        ],
        check_valid=True,
    )
    assert ak.count(array) == 9
    assert ak.to_list(ak.count(array, axis=-1)) == [[3, 0, 2, 1], [], [2, 1]]
    assert ak.to_list(ak.count(array, axis=2)) == [[3, 0, 2, 1], [], [2, 1]]
    assert ak.to_list(ak.count(array, axis=-1, keepdims=True)) == [
        [[3], [0], [2], [1]],
        [],
        [[2], [1]],
    ]
    assert ak.to_list(ak.count(array, axis=-2)) == [[3, 2, 1], [], [2, 1]]
    assert ak.to_list(ak.count(array, axis=1)) == [[3, 2, 1], [], [2, 1]]
    assert ak.to_list(ak.count(array, axis=-2, keepdims=True)) == [
        [[3, 2, 1]],
        [[]],
        [[2, 1]],
    ]


def test_count_nonzero():
    array = ak.Array(
        [
            [
                [np.datetime64("2022"), np.datetime64("2023"), np.datetime64("2025")],
                [],
                [np.datetime64("2027"), np.datetime64("2011")],
                [np.datetime64("2013")],
            ],
            [],
            [[np.datetime64("2017"), np.datetime64("2019")], [np.datetime64("2023")]],
        ],
        check_valid=True,
    )

    assert ak.count_nonzero(array) == 9
    assert ak.to_list(ak.count_nonzero(array, axis=-1)) == [[3, 0, 2, 1], [], [2, 1]]
    assert ak.to_list(ak.count_nonzero(array, axis=-2)) == [[3, 2, 1], [], [2, 1]]

    assert ak.to_list(ak.all(array, axis=-1)) == [
        [True, True, True, True],
        [],
        [True, True],
    ]
    assert ak.to_list(ak.all(array, axis=-2)) == [[True, True, True], [], [True, True]]


#
# def test_argmin_argmax():
#


def test_any_all():
    array = ak.Array(
        [
            [
                [np.datetime64("2022"), np.datetime64("2023"), np.datetime64("2025")],
                [],
                [np.datetime64("2027"), np.datetime64("2011")],
                [np.datetime64("2013")],
            ],
            [],
            [[np.datetime64("2017"), np.datetime64("2019")], [np.datetime64("2023")]],
        ],
        check_valid=True,
    )

    assert ak.to_list(ak.any(array, axis=-1)) == [
        [True, False, True, True],
        [],
        [True, True],
    ]
    assert ak.to_list(ak.any(array, axis=-2)) == [[True, True, True], [], [True, True]]


# def test_prod():
# def test_sum():


def test_min_max():
    array = ak.Array(
        [
            [
                np.datetime64("2020-03-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
                # FIXME: reducer in axis on UnionArray of mixed unit formats
                np.datetime64("2020-05"),
                np.datetime64("2020-01-27T10:41:11"),
                np.datetime64("2020-04-27T10:41:11"),
            ],
            [
                np.datetime64("2020-04-27"),
                np.datetime64("2020-02-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
                np.datetime64("2020-06-27T10:41:11"),
            ],
            [
                np.datetime64("2020-02-27T10:41:11"),
                np.datetime64("2020-03-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
            ],
        ]
    )
    assert ak.to_list(array) == [
        [
            np.datetime64("2020-03-27T10:41:11"),
            np.datetime64("2020-01-27T10:41:11"),
            np.datetime64("2020-05-01T20:56:24"),
            np.datetime64("2020-01-27T10:41:11"),
            np.datetime64("2020-04-27T10:41:11"),
        ],
        [
            np.datetime64("2020-04-27T00:00:00"),
            np.datetime64("2020-02-27T10:41:11"),
            np.datetime64("2020-01-27T10:41:11"),
            np.datetime64("2020-06-27T10:41:11"),
        ],
        [
            np.datetime64("2020-02-27T10:41:11"),
            np.datetime64("2020-03-27T10:41:11"),
            np.datetime64("2020-01-27T10:41:11"),
        ],
    ]
    # FIXME:
    assert ak.min(array) == np.datetime64("2020-01-27T10:41:11")
    # FIXME: assert ak.max(array) == np.datetime64('2020-06-27T10:41:11')
    assert ak.to_list(ak.min(array, axis=0)) == [
        np.datetime64("2020-02-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-04-27T10:41:11"),
    ]
    assert ak.to_list(ak.max(array, axis=0)) == [
        np.datetime64("2020-04-27T00:00:00"),
        np.datetime64("2020-03-27T10:41:11"),
        np.datetime64("2020-05-01T20:56:24"),
        np.datetime64("2020-06-27T10:41:11"),
        np.datetime64("2020-04-27T10:41:11"),
    ]
    assert ak.to_list(ak.min(array, axis=1)) == [
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
    ]
    assert ak.to_list(ak.max(array, axis=1)) == [
        np.datetime64("2020-05-01T20:56:24"),
        np.datetime64("2020-06-27T10:41:11"),
        np.datetime64("2020-03-27T10:41:11"),
    ]


def test_date_time_units():
    array1 = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )
    array2 = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[25s]"
    )
    ak_a1 = ak.Array(array1)
    ak_a2 = ak.Array(array2)
    np_ar1 = ak_a1.to_numpy()
    np_ar2 = ak_a2.to_numpy()

    if np_ar1[0] > np_ar2[0]:
        assert (np_ar1[0] - np.timedelta64(25, "s")) < np_ar2[0]
    else:
        assert (np_ar1[0] + np.timedelta64(25, "s")) >= np_ar2[0]


def test_NumpyArray_date_time():

    dtypes = ["datetime64[s]", "timedelta64[D]"]

    arrays = (np.arange(0, 10, dtype=dtype) for dtype in dtypes)
    for array in arrays:
        print(array)
