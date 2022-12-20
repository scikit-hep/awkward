# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list

primes = [x for x in range(2, 1000) if all(x % n != 0 for n in range(2, x))]


def test():
    data = ak.highlevel.Array([[1, 3, 5, 4, 2], [], [2, 3, 1], [5]]).layout
    assert to_list(ak.min(data, axis=1, initial=4, highlevel=False)) == [1, None, 1, 4]
    assert (
        ak.min(data.to_typetracer(), axis=1, initial=4, highlevel=False).form
        == ak.min(data, axis=1, initial=4, highlevel=False).form
    )
    assert to_list(ak.min(data, axis=1, highlevel=False)) == [1, None, 1, 5]
    assert (
        ak.min(data.to_typetracer(), axis=1, highlevel=False).form
        == ak.min(data, axis=1, highlevel=False).form
    )
    assert to_list(ak.max(data, axis=1, initial=4, highlevel=False)) == [5, None, 4, 5]
    assert (
        ak.max(data.to_typetracer(), axis=1, initial=4, highlevel=False).form
        == ak.max(data, axis=1, initial=4, highlevel=False).form
    )
    assert to_list(ak.max(data, axis=1, highlevel=False)) == [5, None, 3, 5]
    assert (
        ak.max(data.to_typetracer(), axis=1, highlevel=False).form
        == ak.max(data, axis=1, highlevel=False).form
    )

    data = ak.highlevel.Array(
        [[1.1, 3.3, 5.5, 4.4, 2.2], [], [2.2, 3.3, 1.1], [5.5]]
    ).layout
    assert to_list(ak.min(data, axis=1, initial=4, highlevel=False)) == [
        1.1,
        None,
        1.1,
        4,
    ]
    assert (
        ak.min(data.to_typetracer(), axis=1, initial=4, highlevel=False).form
        == ak.min(data, axis=1, initial=4, highlevel=False).form
    )
    assert to_list(ak.min(data, axis=1, highlevel=False)) == [1.1, None, 1.1, 5.5]
    assert (
        ak.min(data.to_typetracer(), axis=1, highlevel=False).form
        == ak.min(data, axis=1, highlevel=False).form
    )
    assert to_list(ak.max(data, axis=1, initial=4, highlevel=False)) == [
        5.5,
        None,
        4,
        5.5,
    ]
    assert (
        ak.max(data.to_typetracer(), axis=1, initial=4, highlevel=False).form
        == ak.max(data, axis=1, initial=4, highlevel=False).form
    )
    assert to_list(ak.max(data, axis=1, highlevel=False)) == [5.5, None, 3.3, 5.5]
    assert (
        ak.max(data.to_typetracer(), axis=1, highlevel=False).form
        == ak.max(data, axis=1, highlevel=False).form
    )

    data = ak.contents.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.int32))
    assert to_list(ak.min(data, axis=-1, initial=4, highlevel=False)) == 4
    assert to_list(ak.min(data, axis=-1, highlevel=False)) == 5
    assert to_list(ak.max(data, axis=-1, initial=114, highlevel=False)) == 114
    assert to_list(ak.max(data, axis=-1, highlevel=False)) == 113

    data = ak.contents.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.uint64))
    assert to_list(ak.min(data, axis=-1, initial=4, highlevel=False)) == 4
    assert to_list(ak.min(data, axis=-1, highlevel=False)) == 5
    assert to_list(ak.max(data, axis=-1, initial=114, highlevel=False)) == 114
    assert to_list(ak.max(data, axis=-1, highlevel=False)) == 113

    data = ak.contents.NumpyArray(np.array(primes[2 : 2 * 3 * 5], dtype=np.uint32))
    assert to_list(ak.min(data, axis=-1, initial=4, highlevel=False)) == 4
    assert to_list(ak.min(data, axis=-1, highlevel=False)) == 5
    assert to_list(ak.max(data, axis=-1, initial=114, highlevel=False)) == 114
    assert to_list(ak.max(data, axis=-1, highlevel=False)) == 113


def test_date_time():
    numpy_array = np.array(
        ["2020-07-27T10:41:11", "2019-01-01", "2020-01-01"], "datetime64[s]"
    )

    array = ak.highlevel.Array(numpy_array)
    assert str(array.type) == "3 * datetime64[s]"
    assert array.to_list() == [
        np.datetime64("2020-07-27T10:41:11"),
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
    ]

    array = array.layout
    assert to_list(array) == [
        np.datetime64("2020-07-27T10:41:11"),
        np.datetime64("2019-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
    ]

    assert ak.max(array, highlevel=False) == numpy_array[0]
    assert ak.min(array, highlevel=False) == numpy_array[1]
