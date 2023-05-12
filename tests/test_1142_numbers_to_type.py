# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_numbers_to_type():
    one, two = (
        ak.highlevel.Array([1, 2, 3]).layout,
        ak.highlevel.Array([4, 5]).layout,
    )

    assert np.asarray(
        ak.values_astype(one, "bool", including_unknown=False)
    ).dtype == np.dtype(np.bool_)
    assert np.asarray(
        ak.values_astype(one, "int8", including_unknown=False)
    ).dtype == np.dtype(np.int8)
    assert np.asarray(
        ak.values_astype(one, "uint8", including_unknown=False)
    ).dtype == np.dtype(np.uint8)
    assert np.asarray(
        ak.values_astype(one, "int16", including_unknown=False)
    ).dtype == np.dtype(np.int16)
    assert np.asarray(
        ak.values_astype(one, "uint16", including_unknown=False)
    ).dtype == np.dtype(np.uint16)
    assert np.asarray(
        ak.values_astype(one, "int32", including_unknown=False)
    ).dtype == np.dtype(np.int32)
    assert np.asarray(
        ak.values_astype(one, "uint32", including_unknown=False)
    ).dtype == np.dtype(np.uint32)
    assert np.asarray(
        ak.values_astype(one, "int64", including_unknown=False)
    ).dtype == np.dtype(np.int64)
    assert np.asarray(
        ak.values_astype(one, "uint64", including_unknown=False)
    ).dtype == np.dtype(np.uint64)
    assert np.asarray(
        ak.values_astype(one, "float32", including_unknown=False)
    ).dtype == np.dtype(np.float32)
    assert np.asarray(
        ak.values_astype(one, "float64", including_unknown=False)
    ).dtype == np.dtype(np.float64)
    assert np.asarray(
        ak.values_astype(one, "complex64", including_unknown=False)
    ).dtype == np.dtype(np.complex64)
    assert np.asarray(
        ak.values_astype(one, "complex128", including_unknown=False)
    ).dtype == np.dtype(np.complex128)
    assert np.asarray(
        ak.values_astype(one, "datetime64", including_unknown=False)
    ).dtype == np.dtype(np.datetime64)
    assert np.asarray(
        ak.values_astype(one, "datetime64[Y]", including_unknown=False)
    ).dtype == np.dtype("datetime64[Y]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[M]", including_unknown=False)
    ).dtype == np.dtype("datetime64[M]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[W]", including_unknown=False)
    ).dtype == np.dtype("datetime64[W]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[D]", including_unknown=False)
    ).dtype == np.dtype("datetime64[D]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[h]", including_unknown=False)
    ).dtype == np.dtype("datetime64[h]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[m]", including_unknown=False)
    ).dtype == np.dtype("datetime64[m]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[s]", including_unknown=False)
    ).dtype == np.dtype("datetime64[s]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[ms]", including_unknown=False)
    ).dtype == np.dtype("datetime64[ms]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[us]", including_unknown=False)
    ).dtype == np.dtype("datetime64[us]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[ns]", including_unknown=False)
    ).dtype == np.dtype("datetime64[ns]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[ps]", including_unknown=False)
    ).dtype == np.dtype("datetime64[ps]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[fs]", including_unknown=False)
    ).dtype == np.dtype("datetime64[fs]")
    assert np.asarray(
        ak.values_astype(one, "datetime64[as]", including_unknown=False)
    ).dtype == np.dtype("datetime64[as]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64", including_unknown=False)
    ).dtype == np.dtype(np.timedelta64)
    assert np.asarray(
        ak.values_astype(one, "timedelta64[Y]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[Y]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[M]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[M]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[W]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[W]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[D]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[D]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[h]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[h]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[m]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[m]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[s]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[s]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[ms]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[ms]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[us]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[us]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[ns]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[ns]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[ps]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[ps]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[fs]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[fs]")
    assert np.asarray(
        ak.values_astype(one, "timedelta64[as]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[as]")

    assert np.asarray(
        ak.values_astype(two, "bool", including_unknown=False)
    ).dtype == np.dtype(np.bool_)
    assert np.asarray(
        ak.values_astype(two, "int8", including_unknown=False)
    ).dtype == np.dtype(np.int8)
    assert np.asarray(
        ak.values_astype(two, "uint8", including_unknown=False)
    ).dtype == np.dtype(np.uint8)
    assert np.asarray(
        ak.values_astype(two, "int16", including_unknown=False)
    ).dtype == np.dtype(np.int16)
    assert np.asarray(
        ak.values_astype(two, "uint16", including_unknown=False)
    ).dtype == np.dtype(np.uint16)
    assert np.asarray(
        ak.values_astype(two, "int32", including_unknown=False)
    ).dtype == np.dtype(np.int32)
    assert np.asarray(
        ak.values_astype(two, "uint32", including_unknown=False)
    ).dtype == np.dtype(np.uint32)
    assert np.asarray(
        ak.values_astype(two, "int64", including_unknown=False)
    ).dtype == np.dtype(np.int64)
    assert np.asarray(
        ak.values_astype(two, "uint64", including_unknown=False)
    ).dtype == np.dtype(np.uint64)
    assert np.asarray(
        ak.values_astype(two, "float32", including_unknown=False)
    ).dtype == np.dtype(np.float32)
    assert np.asarray(
        ak.values_astype(two, "float64", including_unknown=False)
    ).dtype == np.dtype(np.float64)
    assert np.asarray(
        ak.values_astype(two, "complex64", including_unknown=False)
    ).dtype == np.dtype(np.complex64)
    assert np.asarray(
        ak.values_astype(two, "complex128", including_unknown=False)
    ).dtype == np.dtype(np.complex128)
    assert np.asarray(
        ak.values_astype(two, "datetime64", including_unknown=False)
    ).dtype == np.dtype(np.datetime64)
    assert np.asarray(
        ak.values_astype(two, "datetime64[Y]", including_unknown=False)
    ).dtype == np.dtype("datetime64[Y]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[M]", including_unknown=False)
    ).dtype == np.dtype("datetime64[M]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[W]", including_unknown=False)
    ).dtype == np.dtype("datetime64[W]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[D]", including_unknown=False)
    ).dtype == np.dtype("datetime64[D]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[h]", including_unknown=False)
    ).dtype == np.dtype("datetime64[h]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[m]", including_unknown=False)
    ).dtype == np.dtype("datetime64[m]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[s]", including_unknown=False)
    ).dtype == np.dtype("datetime64[s]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[ms]", including_unknown=False)
    ).dtype == np.dtype("datetime64[ms]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[us]", including_unknown=False)
    ).dtype == np.dtype("datetime64[us]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[ns]", including_unknown=False)
    ).dtype == np.dtype("datetime64[ns]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[ps]", including_unknown=False)
    ).dtype == np.dtype("datetime64[ps]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[fs]", including_unknown=False)
    ).dtype == np.dtype("datetime64[fs]")
    assert np.asarray(
        ak.values_astype(two, "datetime64[as]", including_unknown=False)
    ).dtype == np.dtype("datetime64[as]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64", including_unknown=False)
    ).dtype == np.dtype(np.timedelta64)
    assert np.asarray(
        ak.values_astype(two, "timedelta64[Y]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[Y]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[M]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[M]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[W]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[W]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[D]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[D]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[h]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[h]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[m]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[m]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[s]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[s]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[ms]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[ms]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[us]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[us]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[ns]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[ns]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[ps]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[ps]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[fs]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[fs]")
    assert np.asarray(
        ak.values_astype(two, "timedelta64[as]", including_unknown=False)
    ).dtype == np.dtype("timedelta64[as]")
