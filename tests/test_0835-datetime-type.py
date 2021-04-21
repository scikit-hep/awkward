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
    assert (
        str(array.type)
        == '3 * datetime64[parameters={"__array__": "datetime64", "__datetime64_data__": "<M8[s]", "__datetime64_unit__": "1s"}]'
    )
    assert array.tolist() == [
        "2020-07-27T10:41:11",
        "2019-01-01T00:00:00",
        "2020-01-01T00:00:00",
    ]
    for i in range(len(array)):
        assert ak.to_numpy(array)[i] == numpy_array[i]

    date_time = np.datetime64("2020-07-27T10:41:11.200000011", "us")
    array1 = ak.Array(np.array(["2020-07-27T10:41:11.200000011"], "datetime64[us]"))
    assert np.datetime64(array1[0], "us") == date_time

    # FIXME: this prints '2020-07-27T10:41:11.200000'
    print(ak.to_numpy(array1))

    assert ak.to_list(ak.from_iter(ak.to_list(array1))) == [
        "2020-07-27T10:41:11.200000"
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
