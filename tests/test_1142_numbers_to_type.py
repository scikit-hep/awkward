# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_numbers_to_type():
    one, two = (
        ak.highlevel.Array([1, 2, 3]).layout,
        ak.highlevel.Array([4, 5]).layout,
    )

    assert ak.to_numpy(ak._do.numbers_to_type(one, "bool", False)).dtype == np.dtype(
        np.bool_
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "int8", False)).dtype == np.dtype(
        np.int8
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "uint8", False)).dtype == np.dtype(
        np.uint8
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "int16", False)).dtype == np.dtype(
        np.int16
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "uint16", False)).dtype == np.dtype(
        np.uint16
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "int32", False)).dtype == np.dtype(
        np.int32
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "uint32", False)).dtype == np.dtype(
        np.uint32
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "int64", False)).dtype == np.dtype(
        np.int64
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "uint64", False)).dtype == np.dtype(
        np.uint64
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "float32", False)).dtype == np.dtype(
        np.float32
    )
    assert ak.to_numpy(ak._do.numbers_to_type(one, "float64", False)).dtype == np.dtype(
        np.float64
    )
    assert np.asarray(
        ak._do.numbers_to_type(one, "complex64", False).data
    ).dtype == np.dtype(np.complex64)
    assert np.asarray(
        ak._do.numbers_to_type(one, "complex128", False).data
    ).dtype == np.dtype(np.complex128)
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64", False).data
    ).dtype == np.dtype(np.datetime64)
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[Y]", False).data
    ).dtype == np.dtype("datetime64[Y]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[M]", False).data
    ).dtype == np.dtype("datetime64[M]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[W]", False).data
    ).dtype == np.dtype("datetime64[W]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[D]", False).data
    ).dtype == np.dtype("datetime64[D]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[h]", False).data
    ).dtype == np.dtype("datetime64[h]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[m]", False).data
    ).dtype == np.dtype("datetime64[m]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[s]", False).data
    ).dtype == np.dtype("datetime64[s]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[ms]", False).data
    ).dtype == np.dtype("datetime64[ms]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[us]", False).data
    ).dtype == np.dtype("datetime64[us]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[ns]", False).data
    ).dtype == np.dtype("datetime64[ns]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[ps]", False).data
    ).dtype == np.dtype("datetime64[ps]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[fs]", False).data
    ).dtype == np.dtype("datetime64[fs]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "datetime64[as]", False).data
    ).dtype == np.dtype("datetime64[as]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64", False).data
    ).dtype == np.dtype(np.timedelta64)
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[Y]", False).data
    ).dtype == np.dtype("timedelta64[Y]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[M]", False).data
    ).dtype == np.dtype("timedelta64[M]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[W]", False).data
    ).dtype == np.dtype("timedelta64[W]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[D]", False).data
    ).dtype == np.dtype("timedelta64[D]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[h]", False).data
    ).dtype == np.dtype("timedelta64[h]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[m]", False).data
    ).dtype == np.dtype("timedelta64[m]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[s]", False).data
    ).dtype == np.dtype("timedelta64[s]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[ms]", False).data
    ).dtype == np.dtype("timedelta64[ms]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[us]", False).data
    ).dtype == np.dtype("timedelta64[us]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[ns]", False).data
    ).dtype == np.dtype("timedelta64[ns]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[ps]", False).data
    ).dtype == np.dtype("timedelta64[ps]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[fs]", False).data
    ).dtype == np.dtype("timedelta64[fs]")
    assert np.asarray(
        ak._do.numbers_to_type(one, "timedelta64[as]", False).data
    ).dtype == np.dtype("timedelta64[as]")

    assert ak.to_numpy(ak._do.numbers_to_type(two, "bool", False)).dtype == np.dtype(
        np.bool_
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "int8", False)).dtype == np.dtype(
        np.int8
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "uint8", False)).dtype == np.dtype(
        np.uint8
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "int16", False)).dtype == np.dtype(
        np.int16
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "uint16", False)).dtype == np.dtype(
        np.uint16
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "int32", False)).dtype == np.dtype(
        np.int32
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "uint32", False)).dtype == np.dtype(
        np.uint32
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "int64", False)).dtype == np.dtype(
        np.int64
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "uint64", False)).dtype == np.dtype(
        np.uint64
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "float32", False)).dtype == np.dtype(
        np.float32
    )
    assert ak.to_numpy(ak._do.numbers_to_type(two, "float64", False)).dtype == np.dtype(
        np.float64
    )
    assert np.asarray(
        ak._do.numbers_to_type(two, "complex64", False).data
    ).dtype == np.dtype(np.complex64)
    assert np.asarray(
        ak._do.numbers_to_type(two, "complex128", False).data
    ).dtype == np.dtype(np.complex128)
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64", False).data
    ).dtype == np.dtype(np.datetime64)
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[Y]", False).data
    ).dtype == np.dtype("datetime64[Y]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[M]", False).data
    ).dtype == np.dtype("datetime64[M]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[W]", False).data
    ).dtype == np.dtype("datetime64[W]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[D]", False).data
    ).dtype == np.dtype("datetime64[D]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[h]", False).data
    ).dtype == np.dtype("datetime64[h]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[m]", False).data
    ).dtype == np.dtype("datetime64[m]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[s]", False).data
    ).dtype == np.dtype("datetime64[s]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[ms]", False).data
    ).dtype == np.dtype("datetime64[ms]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[us]", False).data
    ).dtype == np.dtype("datetime64[us]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[ns]", False).data
    ).dtype == np.dtype("datetime64[ns]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[ps]", False).data
    ).dtype == np.dtype("datetime64[ps]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[fs]", False).data
    ).dtype == np.dtype("datetime64[fs]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "datetime64[as]", False).data
    ).dtype == np.dtype("datetime64[as]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64", False).data
    ).dtype == np.dtype(np.timedelta64)
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[Y]", False).data
    ).dtype == np.dtype("timedelta64[Y]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[M]", False).data
    ).dtype == np.dtype("timedelta64[M]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[W]", False).data
    ).dtype == np.dtype("timedelta64[W]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[D]", False).data
    ).dtype == np.dtype("timedelta64[D]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[h]", False).data
    ).dtype == np.dtype("timedelta64[h]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[m]", False).data
    ).dtype == np.dtype("timedelta64[m]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[s]", False).data
    ).dtype == np.dtype("timedelta64[s]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[ms]", False).data
    ).dtype == np.dtype("timedelta64[ms]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[us]", False).data
    ).dtype == np.dtype("timedelta64[us]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[ns]", False).data
    ).dtype == np.dtype("timedelta64[ns]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[ps]", False).data
    ).dtype == np.dtype("timedelta64[ps]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[fs]", False).data
    ).dtype == np.dtype("timedelta64[fs]")
    assert np.asarray(
        ak._do.numbers_to_type(two, "timedelta64[as]", False).data
    ).dtype == np.dtype("timedelta64[as]")
