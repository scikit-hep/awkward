# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_values_astype_datetime():
    array1 = ak.values_astype(ak._v2.highlevel.Array([1567416600000]), "datetime64[ms]")
    assert str(array1.type) == "1 * datetime64"
    assert array1.to_list() == [np.datetime64("2019-09-02T09:30:00")]

    array2 = ak.values_astype(
        ak._v2.highlevel.Array([1567416600000]), np.dtype("M8[ms]")
    )
    assert str(array2.type) == "1 * datetime64"
    assert array2.to_list() == [np.datetime64("2019-09-02T09:30:00")]

    array3 = ak.values_astype(
        ak._v2.highlevel.Array([1567416600000000, None]),
        np.datetime64,  # default unit is 'us'
    )
    assert array3.to_list() == [np.datetime64("2019-09-02T09:30:00"), None]


def test_modulo_units():
    array1 = ak.values_astype(
        ak._v2.highlevel.Array([1]), np.dtype("datetime64[100as/1]")
    )
    assert array1.to_list() == [
        np.datetime64("1970-01-01T00:00:00.000000000000000100", "100as")
    ]

    array2 = ak.values_astype(
        ak._v2.highlevel.Array([1]), np.dtype("datetime64[10s/2]")
    )
    assert array2.to_list() == [np.datetime64("1970-01-01T00:00:05.000", "5000ms")]


def test_float_values_astype_datetime():
    array = ak._v2.highlevel.Array([1.9999, 1567416600000, 0, None, 11, 0.555])
    assert str(array.type) == "6 * ?float64"

    dt_array = ak.values_astype(array, "datetime64[ms]")
    assert str(dt_array.type) == "6 * ?datetime64"
    assert dt_array.to_list() == [
        np.datetime64("1970-01-01T00:00:00.001"),
        np.datetime64("2019-09-02T09:30:00.000"),
        np.datetime64("1970-01-01T00:00:00.000"),
        None,
        np.datetime64("1970-01-01T00:00:00.011"),
        np.datetime64("1970-01-01T00:00:00.000"),
    ]
