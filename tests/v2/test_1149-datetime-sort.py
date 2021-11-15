# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

import datetime

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_date_time():
    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )

    array = ak._v2.contents.NumpyArray(numpy_array)
    assert str(array.form.type) == "datetime64[s]"
    assert ak.to_list(array) == [
        np.datetime64("2020-07-27T10:41:11"),
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
    ]
    for i in range(len(array)):
        assert array[i] == numpy_array[i]

    date_time = np.datetime64("2020-07-27T10:41:11.200000011", "us")
    array1 = ak._v2.contents.NumpyArray(
        np.array(["2020-07-27T10:41:11.200000011"], "datetime64[us]")
    )
    assert np.datetime64(array1[0], "us") == date_time

    assert ak.to_list(ak.from_iter(array1)) == [
        np.datetime64("2020-07-27T10:41:11.200000")
    ]


def test_date_time_sort_argsort_unique():
    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )
    array = ak._v2.contents.NumpyArray(numpy_array)
    assert ak.to_list(array.sort()) == [
        datetime.datetime(2019, 1, 1, 0, 0),
        datetime.datetime(2020, 1, 1, 0, 0),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
    ]
    assert ak.to_list(array.argsort()) == [1, 2, 0]
    assert array.is_unique() is True
    assert ak.to_list(array.unique()) == [
        datetime.datetime(2019, 1, 1, 0, 0),
        datetime.datetime(2020, 1, 1, 0, 0),
        datetime.datetime(2020, 7, 27, 10, 41, 11),
    ]


def test_time_delta_sort_argsort_unique():

    numpy_array = np.array(["41", "1", "20"], "timedelta64[D]")

    array = ak.Array(numpy_array)
    array = v1_to_v2(array.layout)
    assert str(array.form.type) == "timedelta64[D]"
    assert ak.to_list(array) == [
        np.timedelta64("41", "D"),
        np.timedelta64("1", "D"),
        np.timedelta64("20", "D"),
    ]
    assert ak.to_list(array.sort()) == [
        datetime.timedelta(days=1),
        datetime.timedelta(days=20),
        datetime.timedelta(days=41),
    ]
    assert ak.to_list(array.argsort()) == [1, 2, 0]
    assert array.is_unique() is True
    assert ak.to_list(array.unique()) == [
        datetime.timedelta(days=1),
        datetime.timedelta(days=20),
        datetime.timedelta(days=41),
    ]
