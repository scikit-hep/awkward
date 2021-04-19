# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_date_time():

    numpy_array = np.array(["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]")

    array = ak.Array(numpy_array)

    assert array.tolist() == ['2020-07-27T10:41:11', '2019-01-01T00:00:00', '2020-01-01T00:00:00']
    for i in range(len(array)):
        assert ak.to_numpy(array)[i] == numpy_array[i]

    date_time = np.datetime64("2020-07-27T10:41:11.200000011", "us")
    array1 = ak.Array(np.array(["2020-07-27T10:41:11.200000011"], "datetime64[us]"))
    assert str(array1.tolist()[0]) == np.datetime_as_string(date_time)

    # FIXME: this is actually '2020-07-27T10:41:11.200000'
    print(ak.to_numpy(array1))

def test_NumpyArray_date_time():

    dtypes = ['datetime64[s]', 'timedelta64[D]']

    arrays = (np.arange(0, 10, dtype=dtype) for dtype in dtypes)
    for array in arrays:
        print(array)
    #raise ValueError
