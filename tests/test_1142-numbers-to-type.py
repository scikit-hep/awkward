# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_numbers_to_type():
    one, two = (
        ak.highlevel.Array([1, 2, 3]).layout,
        ak.highlevel.Array([4, 5]).layout,
    )

    assert np.asarray(ak._do.numbers_to_type(one, "bool")).dtype == np.dtype(np.bool_)
    assert np.asarray(ak._do.numbers_to_type(one, "int8")).dtype == np.dtype(np.int8)
    assert np.asarray(ak._do.numbers_to_type(one, "uint8")).dtype == np.dtype(np.uint8)
    assert np.asarray(ak._do.numbers_to_type(one, "int16")).dtype == np.dtype(np.int16)
    assert np.asarray(ak._do.numbers_to_type(one, "uint16")).dtype == np.dtype(
        np.uint16
    )
    assert np.asarray(ak._do.numbers_to_type(one, "int32")).dtype == np.dtype(np.int32)
    assert np.asarray(ak._do.numbers_to_type(one, "uint32")).dtype == np.dtype(
        np.uint32
    )
    assert np.asarray(ak._do.numbers_to_type(one, "int64")).dtype == np.dtype(np.int64)
    assert np.asarray(ak._do.numbers_to_type(one, "uint64")).dtype == np.dtype(
        np.uint64
    )
    assert np.asarray(ak._do.numbers_to_type(one, "float32")).dtype == np.dtype(
        np.float32
    )
    assert np.asarray(ak._do.numbers_to_type(one, "float64")).dtype == np.dtype(
        np.float64
    )
    assert np.asarray(ak._do.numbers_to_type(one, "complex64")).dtype == np.dtype(
        np.complex64
    )
    assert np.asarray(ak._do.numbers_to_type(one, "complex128")).dtype == np.dtype(
        np.complex128
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64")).dtype == np.dtype(
        np.datetime64
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[Y]")).dtype == np.dtype(
        "datetime64[Y]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[M]")).dtype == np.dtype(
        "datetime64[M]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[W]")).dtype == np.dtype(
        "datetime64[W]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[D]")).dtype == np.dtype(
        "datetime64[D]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[h]")).dtype == np.dtype(
        "datetime64[h]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[m]")).dtype == np.dtype(
        "datetime64[m]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[s]")).dtype == np.dtype(
        "datetime64[s]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[ms]")).dtype == np.dtype(
        "datetime64[ms]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[us]")).dtype == np.dtype(
        "datetime64[us]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[ns]")).dtype == np.dtype(
        "datetime64[ns]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[ps]")).dtype == np.dtype(
        "datetime64[ps]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[fs]")).dtype == np.dtype(
        "datetime64[fs]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "datetime64[as]")).dtype == np.dtype(
        "datetime64[as]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64")).dtype == np.dtype(
        np.timedelta64
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[Y]")).dtype == np.dtype(
        "timedelta64[Y]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[M]")).dtype == np.dtype(
        "timedelta64[M]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[W]")).dtype == np.dtype(
        "timedelta64[W]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[D]")).dtype == np.dtype(
        "timedelta64[D]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[h]")).dtype == np.dtype(
        "timedelta64[h]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[m]")).dtype == np.dtype(
        "timedelta64[m]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[s]")).dtype == np.dtype(
        "timedelta64[s]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[ms]")).dtype == np.dtype(
        "timedelta64[ms]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[us]")).dtype == np.dtype(
        "timedelta64[us]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[ns]")).dtype == np.dtype(
        "timedelta64[ns]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[ps]")).dtype == np.dtype(
        "timedelta64[ps]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[fs]")).dtype == np.dtype(
        "timedelta64[fs]"
    )
    assert np.asarray(ak._do.numbers_to_type(one, "timedelta64[as]")).dtype == np.dtype(
        "timedelta64[as]"
    )

    assert np.asarray(ak._do.numbers_to_type(two, "bool")).dtype == np.dtype(np.bool_)
    assert np.asarray(ak._do.numbers_to_type(two, "int8")).dtype == np.dtype(np.int8)
    assert np.asarray(ak._do.numbers_to_type(two, "uint8")).dtype == np.dtype(np.uint8)
    assert np.asarray(ak._do.numbers_to_type(two, "int16")).dtype == np.dtype(np.int16)
    assert np.asarray(ak._do.numbers_to_type(two, "uint16")).dtype == np.dtype(
        np.uint16
    )
    assert np.asarray(ak._do.numbers_to_type(two, "int32")).dtype == np.dtype(np.int32)
    assert np.asarray(ak._do.numbers_to_type(two, "uint32")).dtype == np.dtype(
        np.uint32
    )
    assert np.asarray(ak._do.numbers_to_type(two, "int64")).dtype == np.dtype(np.int64)
    assert np.asarray(ak._do.numbers_to_type(two, "uint64")).dtype == np.dtype(
        np.uint64
    )
    assert np.asarray(ak._do.numbers_to_type(two, "float32")).dtype == np.dtype(
        np.float32
    )
    assert np.asarray(ak._do.numbers_to_type(two, "float64")).dtype == np.dtype(
        np.float64
    )
    assert np.asarray(ak._do.numbers_to_type(two, "complex64")).dtype == np.dtype(
        np.complex64
    )
    assert np.asarray(ak._do.numbers_to_type(two, "complex128")).dtype == np.dtype(
        np.complex128
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64")).dtype == np.dtype(
        np.datetime64
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[Y]")).dtype == np.dtype(
        "datetime64[Y]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[M]")).dtype == np.dtype(
        "datetime64[M]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[W]")).dtype == np.dtype(
        "datetime64[W]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[D]")).dtype == np.dtype(
        "datetime64[D]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[h]")).dtype == np.dtype(
        "datetime64[h]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[m]")).dtype == np.dtype(
        "datetime64[m]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[s]")).dtype == np.dtype(
        "datetime64[s]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[ms]")).dtype == np.dtype(
        "datetime64[ms]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[us]")).dtype == np.dtype(
        "datetime64[us]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[ns]")).dtype == np.dtype(
        "datetime64[ns]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[ps]")).dtype == np.dtype(
        "datetime64[ps]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[fs]")).dtype == np.dtype(
        "datetime64[fs]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "datetime64[as]")).dtype == np.dtype(
        "datetime64[as]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64")).dtype == np.dtype(
        np.timedelta64
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[Y]")).dtype == np.dtype(
        "timedelta64[Y]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[M]")).dtype == np.dtype(
        "timedelta64[M]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[W]")).dtype == np.dtype(
        "timedelta64[W]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[D]")).dtype == np.dtype(
        "timedelta64[D]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[h]")).dtype == np.dtype(
        "timedelta64[h]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[m]")).dtype == np.dtype(
        "timedelta64[m]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[s]")).dtype == np.dtype(
        "timedelta64[s]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[ms]")).dtype == np.dtype(
        "timedelta64[ms]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[us]")).dtype == np.dtype(
        "timedelta64[us]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[ns]")).dtype == np.dtype(
        "timedelta64[ns]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[ps]")).dtype == np.dtype(
        "timedelta64[ps]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[fs]")).dtype == np.dtype(
        "timedelta64[fs]"
    )
    assert np.asarray(ak._do.numbers_to_type(two, "timedelta64[as]")).dtype == np.dtype(
        "timedelta64[as]"
    )
