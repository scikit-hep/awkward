# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import datetime

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_date_time():
    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )

    array = ak.contents.NumpyArray(numpy_array)
    assert str(array.form.type) == "datetime64[s]"
    assert to_list(array) == [
        np.datetime64("2020-07-27T10:41:11"),
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
    ]
    for i in range(len(array)):
        assert array[i] == numpy_array[i]

    date_time = np.datetime64("2020-07-27T10:41:11.200000011", "us")
    array1 = ak.contents.NumpyArray(
        np.array(["2020-07-27T10:41:11.200000011"], "datetime64[us]")
    )
    assert np.datetime64(array1[0], "us") == date_time

    assert to_list(ak.operations.from_iter(array1)) == [
        datetime.datetime.fromisoformat("2020-07-27T10:41:11.200000")
    ]


def test_date_time_minmax():
    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )
    array = ak.contents.NumpyArray(numpy_array)
    assert ak.max(array, axis=-1, highlevel=False) == numpy_array[0]
    assert ak.min(array, axis=-1, highlevel=False) == numpy_array[1]


def test_time_delta():

    numpy_array = np.array(["41", "1", "20"], "timedelta64[D]")

    array = ak.highlevel.Array(numpy_array).layout
    assert str(array.form.type) == "timedelta64[D]"
    assert to_list(array) == [
        np.timedelta64("41", "D"),
        np.timedelta64("1", "D"),
        np.timedelta64("20", "D"),
    ]
    for i in range(len(array)):
        assert array[i] == numpy_array[i]


def test_datetime64_ArrayBuilder():
    builder = ak.highlevel.ArrayBuilder()
    dt = np.datetime64("2020-03-27T10:41:12", "25us")
    dt1 = np.datetime64("2020-03-27T10:41", "15s")
    dt2 = np.datetime64("2020-05")
    builder.datetime(dt1)
    builder.datetime("2020-03-27T10:41:11")
    builder.datetime(dt)
    builder.datetime("2021-03-27")
    builder.datetime("2020-03-27T10:41:13")
    builder.datetime(dt2)
    builder.datetime("2020-05-01T00:00:00.000000")
    builder.datetime("2020-07-27T10:41:11.200000")

    assert to_list(builder.snapshot()) == [
        np.datetime64("2020-03-27T10:41:00", "15s"),
        np.datetime64("2020-03-27T10:41:11"),
        np.datetime64("2020-03-27T10:41:12.000000", "25us"),
        np.datetime64("2021-03-27"),
        np.datetime64("2020-03-27T10:41:13"),
        np.datetime64("2020-05"),
        np.datetime64("2020-05-01T00:00:00.000000"),
        np.datetime64("2020-07-27T10:41:11.200000"),
    ]


def test_highlevel_datetime64_ArrayBuilder():
    builder = ak.highlevel.ArrayBuilder()
    dt = np.datetime64("2020-03-27T10:41:12", "25us")
    dt1 = np.datetime64("2020-03-27T10:41", "15s")
    dt2 = np.datetime64("2020-05")
    builder.datetime(dt1)
    builder.datetime("2020-03-27T10:41:11")
    builder.datetime(dt)
    builder.datetime("2021-03-27")
    builder.datetime("2020-03-27T10:41:13")
    builder.timedelta(np.timedelta64(5, "s"))
    builder.datetime(dt2)
    builder.datetime("2020-05-01T00:00:00.000000")
    builder.datetime("2020-07-27T10:41:11.200000")
    builder.integer(1)
    builder.timedelta(np.timedelta64(5, "s"))

    assert to_list(builder.snapshot()) == [
        np.datetime64("2020-03-27T10:41", "15s"),
        np.datetime64("2020-03-27T10:41:11"),
        np.datetime64("2020-03-27T10:41:12", "25us"),
        np.datetime64("2021-03-27"),
        np.datetime64("2020-03-27T10:41:13"),
        np.timedelta64(5, "s"),
        np.datetime64("2020-05"),
        np.datetime64("2020-05-01T00:00:00.000000"),
        np.datetime64("2020-07-27T10:41:11.200000"),
        1,
        np.timedelta64(5, "s"),
    ]


def test_timedelta64_ArrayBuilder():
    builder = ak.highlevel.ArrayBuilder()
    builder.timedelta(np.timedelta64(5, "W"))
    builder.timedelta(np.timedelta64(5, "D"))
    builder.timedelta(np.timedelta64(5, "s"))

    assert to_list(builder.snapshot()) == [
        datetime.timedelta(weeks=5),
        datetime.timedelta(5),
        datetime.timedelta(0, 5),
    ]


def test_timedelta64_ArrayBuilder_py3():
    builder = ak.highlevel.ArrayBuilder()
    builder.timedelta(np.timedelta64(5, "W"))
    builder.timedelta(np.timedelta64(5, "D"))
    builder.timedelta(np.timedelta64(5, "s"))

    assert to_list(builder.snapshot()) == [
        np.timedelta64(5 * 7 * 24 * 60 * 60, "s"),
        np.timedelta64(5 * 24 * 60 * 60, "s"),
        np.timedelta64(5, "s"),
    ]


def test_highlevel_timedelta64_ArrayBuilder():
    builder = ak.highlevel.ArrayBuilder()
    builder.timedelta(np.timedelta64(5, "W"))
    builder.timedelta(np.timedelta64(5, "D"))
    builder.timedelta(np.timedelta64(5, "s"))
    builder.integer(1)
    builder.datetime("2020-05-01T00:00:00.000000")

    assert to_list(builder.snapshot()) == [
        np.timedelta64(5 * 7 * 24 * 60 * 60, "s"),
        np.timedelta64(5 * 24 * 60 * 60, "s"),
        np.timedelta64(5, "s"),
        1,
        np.datetime64("2020-05-01T00:00:00.000000"),
    ]


def test_count_axis_None():
    array = ak.highlevel.Array(
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
    assert ak.operations.count(array) == 9


def test_count():
    array = ak.highlevel.Array(
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
    assert to_list(ak.operations.count(array, axis=-1)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=2)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=-1, keepdims=True)) == [
        [[3], [0], [2], [1]],
        [],
        [[2], [1]],
    ]
    assert to_list(ak.operations.count(array, axis=-2)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=1)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count(array, axis=-2, keepdims=True)) == [
        [[3, 2, 1]],
        [[]],
        [[2, 1]],
    ]


def test_count_nonzeroaxis_None():
    array = ak.highlevel.Array(
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
    assert ak.operations.count_nonzero(array) == 9


def test_count_nonzero():
    array = ak.highlevel.Array(
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
    assert to_list(ak.operations.count_nonzero(array, axis=-1)) == [
        [3, 0, 2, 1],
        [],
        [2, 1],
    ]
    assert to_list(ak.operations.count_nonzero(array, axis=-2)) == [
        [3, 2, 1],
        [],
        [2, 1],
    ]


def test_all_nonzero():
    array = ak.highlevel.Array(
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

    assert to_list(ak.operations.all(array, axis=-1)) == [
        [True, True, True, True],
        [],
        [True, True],
    ]
    assert to_list(ak.operations.all(array, axis=-2)) == [
        [True, True, True],
        [],
        [True, True],
    ]


def test_argmin_argmax_axis_None():
    array = ak.highlevel.Array(
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
    assert ak.operations.argmin(array) == 4
    assert ak.operations.argmax(array) == 3


def test_argmin_argmax():
    array = ak.highlevel.Array(
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
    assert to_list(ak.operations.argmin(array, axis=0)) == [
        [1, 1, 0],
        [1],
        [0, 0],
        [0],
    ]
    assert to_list(ak.operations.argmax(array, axis=0)) == [
        [0, 0, 0],
        [1],
        [0, 0],
        [0],
    ]
    assert to_list(ak.operations.argmin(array, axis=1)) == [
        [3, 2, 0],
        [],
        [0, 0],
    ]
    assert to_list(ak.operations.argmax(array, axis=1)) == [
        [2, 0, 0],
        [],
        [1, 0],
    ]

    array = ak.operations.from_iter(
        [
            [
                [
                    np.datetime64("2021-01-20"),
                    np.datetime64("2021-01-10"),
                    np.datetime64("2021-01-30"),
                ]
            ],
            [[]],
            [None, None, None],
            [
                [
                    np.datetime64("2021-01-14"),
                    np.datetime64("2021-01-15"),
                    np.datetime64("2021-01-16"),
                ]
            ],
        ],
        highlevel=False,
    )
    assert to_list(ak.argmin(array, axis=2, highlevel=False)) == [
        [1],
        [None],
        [None, None, None],
        [0],
    ]


def test_any_all():
    array = ak.highlevel.Array(
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

    assert to_list(ak.operations.any(array, axis=-1)) == [
        [True, False, True, True],
        [],
        [True, True],
    ]
    assert to_list(ak.operations.any(array, axis=-2)) == [
        [True, True, True],
        [],
        [True, True],
    ]


def test_prod():
    array = ak.highlevel.Array(
        np.array(["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]")
    )
    with pytest.raises(ValueError):
        ak.operations.prod(array, axis=-1)


def test_min_max():
    array = ak.highlevel.Array(
        [
            [
                np.datetime64("2020-03-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
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
    ).layout

    assert to_list(array) == [
        [
            datetime.datetime(2020, 3, 27, 10, 41, 11),
            datetime.datetime(2020, 1, 27, 10, 41, 11),
            datetime.date(2020, 5, 1),
            datetime.datetime(2020, 1, 27, 10, 41, 11),
            datetime.datetime(2020, 4, 27, 10, 41, 11),
        ],
        [
            datetime.date(2020, 4, 27),
            datetime.datetime(2020, 2, 27, 10, 41, 11),
            datetime.datetime(2020, 1, 27, 10, 41, 11),
            datetime.datetime(2020, 6, 27, 10, 41, 11),
        ],
        [
            datetime.datetime(2020, 2, 27, 10, 41, 11),
            datetime.datetime(2020, 3, 27, 10, 41, 11),
            datetime.datetime(2020, 1, 27, 10, 41, 11),
        ],
    ]

    array = ak.highlevel.Array(
        [
            [
                np.datetime64("2020-03-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
                np.datetime64("2020-05-01T00:00:00"),
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
    ).layout
    assert to_list(ak.min(array, axis=-1, highlevel=False)) == [
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
    ]
    assert to_list(ak.max(array, axis=-1, highlevel=False)) == [
        datetime.datetime(2020, 5, 1, 0, 0),
        datetime.datetime(2020, 6, 27, 10, 41, 11),
        datetime.datetime(2020, 3, 27, 10, 41, 11),
    ]


def test_highlevel_min_max_axis_None():
    array = ak.highlevel.Array(
        [
            [
                np.datetime64("2020-03-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
                np.datetime64("2020-05-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
                np.datetime64("2020-04-27T10:41:11"),
            ],
            [
                np.datetime64("2020-04-27T10:41:11"),
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
    assert to_list(array) == [
        [
            np.datetime64("2020-03-27T10:41:11"),
            np.datetime64("2020-01-27T10:41:11"),
            np.datetime64("2020-05-27T10:41:11"),
            np.datetime64("2020-01-27T10:41:11"),
            np.datetime64("2020-04-27T10:41:11"),
        ],
        [
            np.datetime64("2020-04-27T10:41:11"),
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
    assert ak.operations.min(array) == np.datetime64("2020-01-27T10:41:11")
    assert ak.operations.max(array) == np.datetime64("2020-06-27T10:41:11")


def test_highlevel_min_max():
    array = ak.highlevel.Array(
        [
            [
                np.datetime64("2020-03-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
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
    assert to_list(array) == [
        [
            np.datetime64("2020-03-27T10:41:11"),
            np.datetime64("2020-01-27T10:41:11"),
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

    array = ak.highlevel.Array(
        [
            [
                np.datetime64("2020-03-27T10:41:11"),
                np.datetime64("2020-01-27T10:41:11"),
                np.datetime64("2020-05-01T00:00:00"),
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
    )
    assert to_list(ak.operations.min(array, axis=0)) == [
        np.datetime64("2020-02-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-04-27T10:41:11"),
    ]
    assert to_list(ak.operations.max(array, axis=0)) == [
        datetime.datetime(2020, 4, 27, 0, 0),
        datetime.datetime(2020, 3, 27, 10, 41, 11),
        datetime.datetime(2020, 5, 1, 0, 0),
        datetime.datetime(2020, 6, 27, 10, 41, 11),
        datetime.datetime(2020, 4, 27, 10, 41, 11),
    ]
    assert to_list(ak.operations.min(array, axis=1)) == [
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
        np.datetime64("2020-01-27T10:41:11"),
    ]
    assert to_list(ak.operations.max(array, axis=1)) == [
        datetime.datetime(2020, 5, 1, 0, 0),
        datetime.datetime(2020, 6, 27, 10, 41, 11),
        datetime.datetime(2020, 3, 27, 10, 41, 11),
    ]


def test_date_time_units():
    array1 = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )
    array2 = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[25s]"
    )
    ak_a1 = ak.highlevel.Array(array1).layout

    ak_a2 = ak.highlevel.Array(array2).layout

    np_ar1 = ak.operations.to_numpy(ak_a1)
    np_ar2 = ak.operations.to_numpy(ak_a2)

    if np_ar1[0] > np_ar2[0]:
        assert (np_ar1[0] - np.timedelta64(25, "s")) < np_ar2[0]
    else:
        assert (np_ar1[0] + np.timedelta64(25, "s")) >= np_ar2[0]


def test_sum():

    dtypes = ["datetime64[s]", "timedelta64[D]"]

    arrays = (np.arange(0, 12, dtype=dtype) for dtype in dtypes)
    for array in arrays:
        content = ak.contents.NumpyArray(array)
        offsets = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
        depth = ak.contents.ListOffsetArray(offsets, content)

        if np.issubdtype(array.dtype, np.timedelta64):
            assert to_list(ak.sum(depth, -1, highlevel=False)) == [
                datetime.timedelta(6),
                datetime.timedelta(22),
                datetime.timedelta(38),
            ]

            assert to_list(ak.sum(depth, 1, highlevel=False)) == [
                datetime.timedelta(6),
                datetime.timedelta(22),
                datetime.timedelta(38),
            ]

            assert to_list(ak.sum(depth, -2, highlevel=False)) == [
                datetime.timedelta(12),
                datetime.timedelta(15),
                datetime.timedelta(18),
                datetime.timedelta(21),
            ]
            assert to_list(ak.sum(depth, 0, highlevel=False)) == [
                datetime.timedelta(12),
                datetime.timedelta(15),
                datetime.timedelta(18),
                datetime.timedelta(21),
            ]

        else:
            with pytest.raises(ValueError):
                ak.sum(depth, -1, highlevel=False)


def test_more():
    nparray = np.array(
        [np.datetime64("2021-06-03T10:00"), np.datetime64("2021-06-03T11:00")]
    )
    akarray = ak.highlevel.Array(nparray)

    assert (akarray[1:] - akarray[:-1]).to_list() == [np.timedelta64(60, "m")]
    assert ak.operations.sum(akarray[1:] - akarray[:-1]) == np.timedelta64(60, "m")
    assert ak.operations.sum(akarray[1:] - akarray[:-1], axis=0) == [
        np.timedelta64(60, "m")
    ]


def test_ufunc_sum():
    nparray = np.array(
        [np.datetime64("2021-06-03T10:00"), np.datetime64("2021-06-03T11:00")]
    )
    akarray = ak.highlevel.Array(nparray)

    with pytest.raises(TypeError):
        akarray[1:] + akarray[:-1]


def test_ufunc_mul():
    nparray = np.array(
        [np.datetime64("2021-06-03T10:00"), np.datetime64("2021-06-03T11:00")]
    )
    akarray = ak.highlevel.Array(nparray)

    with pytest.raises(TypeError):
        akarray * 2

    assert ak.highlevel.Array([np.timedelta64(3, "D")])[0] == np.timedelta64(3, "D")


def test_NumpyArray_layout_as_objects():
    with pytest.raises(TypeError):
        ak.highlevel.Array(
            ak.contents.NumpyArray(
                ["2019-09-02T09:30:00", "2019-09-13T09:30:00", "2019-09-21T20:00:00"]
            )
        )

    array0 = ak.contents.NumpyArray(
        np.array(
            ["2019-09-02T09:30:00", "2019-09-13T09:30:00", "2019-09-21T20:00:00"],
            dtype=np.datetime64,
        )
    )

    assert to_list(array0) == [
        datetime.datetime(2019, 9, 2, 9, 30),
        datetime.datetime(2019, 9, 13, 9, 30),
        datetime.datetime(2019, 9, 21, 20, 0),
    ]


def test_NumpyArray_layout():
    array = ak.contents.NumpyArray(
        [
            np.datetime64("2019-09-02T09:30:00"),
            np.datetime64("2019-09-13T09:30:00"),
            np.datetime64("2019-09-21T20:00:00"),
        ]
    )

    assert to_list(array) == [
        np.datetime64("2019-09-02T09:30:00"),
        np.datetime64("2019-09-13T09:30:00"),
        np.datetime64("2019-09-21T20:00:00"),
    ]
