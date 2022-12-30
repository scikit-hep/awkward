# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import datetime

import numpy as np
import pytest  # noqa: F401

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
        np.datetime64("2020-07-27T10:41:11.200000")
    ]


def test_date_time_sort_argsort_unique():
    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )
    array = ak.contents.NumpyArray(numpy_array)
    assert to_list(ak.sort(array, highlevel=False)) == [
        datetime.datetime(2019, 1, 1, 0, 0),
        datetime.datetime(2020, 1, 1, 0, 0),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
    ]
    assert to_list(ak.argsort(array, highlevel=False)) == [1, 2, 0]
    assert ak._do.is_unique(array) is True
    assert to_list(ak._do.unique(array)) == [
        datetime.datetime(2019, 1, 1, 0, 0),
        datetime.datetime(2020, 1, 1, 0, 0),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
    ]


def test_time_delta_sort_argsort_unique():

    numpy_array = np.array(["41", "1", "20"], "timedelta64[D]")

    array = ak.highlevel.Array(numpy_array).layout
    assert str(array.form.type) == "timedelta64[D]"
    assert to_list(array) == [
        np.timedelta64("41", "D"),
        np.timedelta64("1", "D"),
        np.timedelta64("20", "D"),
    ]
    assert to_list(ak.sort(array, highlevel=False)) == [
        datetime.timedelta(days=1),
        datetime.timedelta(days=20),
        datetime.timedelta(days=41),
    ]
    assert to_list(ak.argsort(array, highlevel=False)) == [1, 2, 0]
    assert ak._do.is_unique(array) is True
    assert to_list(ak._do.unique(array)) == [
        datetime.timedelta(days=1),
        datetime.timedelta(days=20),
        datetime.timedelta(days=41),
    ]
