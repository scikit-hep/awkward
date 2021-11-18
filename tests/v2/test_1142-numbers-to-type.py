# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_numbers_to_type():
    one, two, three = (
        ak.Array([1, 2, 3]).layout,
        ak.Array([4, 5]).layout,
        ak.Array([6]).layout,
    )
    one = v1_to_v2(one)
    two = v1_to_v2(two)
    three = v1_to_v2(three)

    assert np.asarray(one.numbers_to_type("bool")).dtype == np.dtype(np.bool_)
    assert np.asarray(one.numbers_to_type("int8")).dtype == np.dtype(np.int8)
    assert np.asarray(one.numbers_to_type("uint8")).dtype == np.dtype(np.uint8)
    assert np.asarray(one.numbers_to_type("int16")).dtype == np.dtype(np.int16)
    assert np.asarray(one.numbers_to_type("uint16")).dtype == np.dtype(np.uint16)
    assert np.asarray(one.numbers_to_type("int32")).dtype == np.dtype(np.int32)
    assert np.asarray(one.numbers_to_type("uint32")).dtype == np.dtype(np.uint32)
    assert np.asarray(one.numbers_to_type("int64")).dtype == np.dtype(np.int64)
    assert np.asarray(one.numbers_to_type("uint64")).dtype == np.dtype(np.uint64)
    assert np.asarray(one.numbers_to_type("float32")).dtype == np.dtype(np.float32)
    assert np.asarray(one.numbers_to_type("float64")).dtype == np.dtype(np.float64)
    assert np.asarray(one.numbers_to_type("complex64")).dtype == np.dtype(np.complex64)
    assert np.asarray(one.numbers_to_type("complex128")).dtype == np.dtype(
        np.complex128
    )
    assert np.asarray(one.numbers_to_type("datetime64")).dtype == np.dtype(
        np.datetime64
    )
    assert np.asarray(one.numbers_to_type("datetime64[Y]")).dtype == np.dtype(
        "datetime64[Y]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[M]")).dtype == np.dtype(
        "datetime64[M]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[W]")).dtype == np.dtype(
        "datetime64[W]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[D]")).dtype == np.dtype(
        "datetime64[D]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[h]")).dtype == np.dtype(
        "datetime64[h]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[m]")).dtype == np.dtype(
        "datetime64[m]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[s]")).dtype == np.dtype(
        "datetime64[s]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[ms]")).dtype == np.dtype(
        "datetime64[ms]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[us]")).dtype == np.dtype(
        "datetime64[us]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[ns]")).dtype == np.dtype(
        "datetime64[ns]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[ps]")).dtype == np.dtype(
        "datetime64[ps]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[fs]")).dtype == np.dtype(
        "datetime64[fs]"
    )
    assert np.asarray(one.numbers_to_type("datetime64[as]")).dtype == np.dtype(
        "datetime64[as]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64")).dtype == np.dtype(
        np.timedelta64
    )
    assert np.asarray(one.numbers_to_type("timedelta64[Y]")).dtype == np.dtype(
        "timedelta64[Y]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[M]")).dtype == np.dtype(
        "timedelta64[M]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[W]")).dtype == np.dtype(
        "timedelta64[W]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[D]")).dtype == np.dtype(
        "timedelta64[D]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[h]")).dtype == np.dtype(
        "timedelta64[h]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[m]")).dtype == np.dtype(
        "timedelta64[m]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[s]")).dtype == np.dtype(
        "timedelta64[s]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[ms]")).dtype == np.dtype(
        "timedelta64[ms]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[us]")).dtype == np.dtype(
        "timedelta64[us]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[ns]")).dtype == np.dtype(
        "timedelta64[ns]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[ps]")).dtype == np.dtype(
        "timedelta64[ps]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[fs]")).dtype == np.dtype(
        "timedelta64[fs]"
    )
    assert np.asarray(one.numbers_to_type("timedelta64[as]")).dtype == np.dtype(
        "timedelta64[as]"
    )

    assert np.asarray(two.numbers_to_type("bool")).dtype == np.dtype(np.bool_)
    assert np.asarray(two.numbers_to_type("int8")).dtype == np.dtype(np.int8)
    assert np.asarray(two.numbers_to_type("uint8")).dtype == np.dtype(np.uint8)
    assert np.asarray(two.numbers_to_type("int16")).dtype == np.dtype(np.int16)
    assert np.asarray(two.numbers_to_type("uint16")).dtype == np.dtype(np.uint16)
    assert np.asarray(two.numbers_to_type("int32")).dtype == np.dtype(np.int32)
    assert np.asarray(two.numbers_to_type("uint32")).dtype == np.dtype(np.uint32)
    assert np.asarray(two.numbers_to_type("int64")).dtype == np.dtype(np.int64)
    assert np.asarray(two.numbers_to_type("uint64")).dtype == np.dtype(np.uint64)
    assert np.asarray(two.numbers_to_type("float32")).dtype == np.dtype(np.float32)
    assert np.asarray(two.numbers_to_type("float64")).dtype == np.dtype(np.float64)
    assert np.asarray(two.numbers_to_type("complex64")).dtype == np.dtype(np.complex64)
    assert np.asarray(two.numbers_to_type("complex128")).dtype == np.dtype(
        np.complex128
    )
    assert np.asarray(two.numbers_to_type("datetime64")).dtype == np.dtype(
        np.datetime64
    )
    assert np.asarray(two.numbers_to_type("datetime64[Y]")).dtype == np.dtype(
        "datetime64[Y]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[M]")).dtype == np.dtype(
        "datetime64[M]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[W]")).dtype == np.dtype(
        "datetime64[W]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[D]")).dtype == np.dtype(
        "datetime64[D]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[h]")).dtype == np.dtype(
        "datetime64[h]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[m]")).dtype == np.dtype(
        "datetime64[m]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[s]")).dtype == np.dtype(
        "datetime64[s]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[ms]")).dtype == np.dtype(
        "datetime64[ms]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[us]")).dtype == np.dtype(
        "datetime64[us]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[ns]")).dtype == np.dtype(
        "datetime64[ns]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[ps]")).dtype == np.dtype(
        "datetime64[ps]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[fs]")).dtype == np.dtype(
        "datetime64[fs]"
    )
    assert np.asarray(two.numbers_to_type("datetime64[as]")).dtype == np.dtype(
        "datetime64[as]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64")).dtype == np.dtype(
        np.timedelta64
    )
    assert np.asarray(two.numbers_to_type("timedelta64[Y]")).dtype == np.dtype(
        "timedelta64[Y]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[M]")).dtype == np.dtype(
        "timedelta64[M]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[W]")).dtype == np.dtype(
        "timedelta64[W]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[D]")).dtype == np.dtype(
        "timedelta64[D]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[h]")).dtype == np.dtype(
        "timedelta64[h]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[m]")).dtype == np.dtype(
        "timedelta64[m]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[s]")).dtype == np.dtype(
        "timedelta64[s]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[ms]")).dtype == np.dtype(
        "timedelta64[ms]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[us]")).dtype == np.dtype(
        "timedelta64[us]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[ns]")).dtype == np.dtype(
        "timedelta64[ns]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[ps]")).dtype == np.dtype(
        "timedelta64[ps]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[fs]")).dtype == np.dtype(
        "timedelta64[fs]"
    )
    assert np.asarray(two.numbers_to_type("timedelta64[as]")).dtype == np.dtype(
        "timedelta64[as]"
    )
