# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]


def test():
    data = ak.Array([[1, 3, 5, 4, 2], [], [2, 3, 1], [5]])
    data = v1_to_v2(data.layout)
    assert ak.to_list(data.min(axis=1, initial=4)) == [1, None, 1, 4]
    assert (
        data.typetracer.min(axis=1, initial=4).form == data.min(axis=1, initial=4).form
    )
    assert ak.to_list(data.min(axis=1)) == [1, None, 1, 5]
    assert data.typetracer.min(axis=1).form == data.min(axis=1).form
    assert ak.to_list(data.max(axis=1, initial=4)) == [5, None, 4, 5]
    assert (
        data.typetracer.max(axis=1, initial=4).form == data.max(axis=1, initial=4).form
    )
    assert ak.to_list(data.max(axis=1)) == [5, None, 3, 5]
    assert data.typetracer.max(axis=1).form == data.max(axis=1).form

    data = ak.Array([[1.1, 3.3, 5.5, 4.4, 2.2], [], [2.2, 3.3, 1.1], [5.5]])
    data = v1_to_v2(data.layout)
    assert ak.to_list(data.min(axis=1, initial=4)) == [1.1, None, 1.1, 4]
    assert (
        data.typetracer.min(axis=1, initial=4).form == data.min(axis=1, initial=4).form
    )
    assert ak.to_list(data.min(axis=1)) == [1.1, None, 1.1, 5.5]
    assert data.typetracer.min(axis=1).form == data.min(axis=1).form
    assert ak.to_list(data.max(axis=1, initial=4)) == [5.5, None, 4, 5.5]
    assert (
        data.typetracer.max(axis=1, initial=4).form == data.max(axis=1, initial=4).form
    )
    assert ak.to_list(data.max(axis=1)) == [5.5, None, 3.3, 5.5]
    assert data.typetracer.max(axis=1).form == data.max(axis=1).form

    array = ak.layout.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.int32))
    array = data = v1_to_v2(array)
    assert ak.to_list(data.min(axis=-1, initial=4)) == 4
    assert ak.to_list(data.min(axis=-1)) == 5
    assert ak.to_list(data.max(axis=-1, initial=114)) == 114
    assert ak.to_list(data.max(axis=-1)) == 113

    array = ak.layout.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.uint64))
    array = data = v1_to_v2(array)
    assert ak.to_list(data.min(axis=-1, initial=4)) == 4
    assert ak.to_list(data.min(axis=-1)) == 5
    assert ak.to_list(data.max(axis=-1, initial=114)) == 114
    assert ak.to_list(data.max(axis=-1)) == 113

    array = ak.layout.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.uint32))
    array = data = v1_to_v2(array)
    assert ak.to_list(data.min(axis=-1, initial=4)) == 4
    assert ak.to_list(data.min(axis=-1)) == 5
    assert ak.to_list(data.max(axis=-1, initial=114)) == 114
    assert ak.to_list(data.max(axis=-1)) == 113


def test_date_time():
    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )

    array = ak.Array(numpy_array)
    assert str(array.type) == "3 * datetime64"
    assert array.tolist() == [
        np.datetime64("2020-07-27T10:41:11"),
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
    ]

    array = v1_to_v2(array.layout)
    assert ak.to_list(array) == [
        np.datetime64("2020-07-27T10:41:11"),
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
    ]

    assert array.max() == numpy_array[0]
    assert array.min() == numpy_array[1]
