# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_dtype():
    assert not ak.almost_equal(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int8),
        dtype_exact=True,
    )
    assert ak.almost_equal(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int8),
        dtype_exact=False,
    )
    assert not ak.almost_equal([1, 2, 3], ["1", "2", "3"], dtype_exact=False)
    assert not ak.almost_equal([1, 2, 3], ["1", "2", "3"], dtype_exact=True)
    assert not ak.almost_equal(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.timedelta64),
        dtype_exact=True,
    )
    assert not ak.almost_equal(
        np.array([1, 2, 3], dtype=np.dtype("<M8[D]")),
        np.array([1, 2, 3], dtype=np.dtype("<m8[D]")),
        dtype_exact=True,
    )
    assert not ak.almost_equal(
        np.array([1, 2, 3], dtype=np.dtype("<M8[D]")),
        np.array([1, 2, 3], dtype=np.dtype("<m8[D]")),
        dtype_exact=False,
    )


def test_regular():
    array = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert ak.almost_equal(array, array, check_regular=True)
    assert ak.almost_equal(array, array, check_regular=False)

    assert not ak.almost_equal(array, ak.to_regular(array), check_regular=True)
    assert ak.almost_equal(array, ak.to_regular(array), check_regular=False)

    # Strided to product `ListArray`
    list_array = array[::1]
    assert isinstance(list_array.layout, ak.contents.ListArray)
    assert ak.almost_equal(array, list_array, check_regular=False)
    assert ak.almost_equal(array, list_array, check_regular=True)

    assert not ak.almost_equal(list_array, ak.to_regular(array), check_regular=True)
    assert ak.almost_equal(list_array, ak.to_regular(array), check_regular=False)

    numpy_array = ak.from_numpy(ak.to_numpy(array), regulararray=False)
    assert ak.almost_equal(numpy_array, ak.to_regular(array), check_regular=True)
    assert ak.almost_equal(numpy_array, ak.to_regular(array), check_regular=False)
    assert not ak.almost_equal(numpy_array, array, check_regular=True)
    assert ak.almost_equal(numpy_array, array, check_regular=False)


def test_parameters():
    array = ak.with_parameter([1, 2, 3], "name", "Bob Dylan")
    assert not ak.almost_equal(array, [1, 2, 3])
    assert ak.almost_equal(array, [1, 2, 3], check_parameters=False)

    array_other = ak.with_parameter(array, "name", "Emmy Noether")
    assert not ak.almost_equal(array, array_other)
    assert ak.almost_equal(array, array_other, check_parameters=False)


def test_option():
    array = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    masked_array = array.mask[[True, False, False]]
    alternative_array = ak.Array([[1, 2, 3], [9, 10, 11], [12, 13, 14]])
    alternative_masked_array = alternative_array.mask[[True, False, False]]

    assert isinstance(masked_array.layout, ak.contents.ByteMaskedArray)
    assert not ak.almost_equal(array, alternative_array)
    assert ak.almost_equal(masked_array, alternative_masked_array)

    masked_array_indexed = masked_array.layout.to_IndexedOptionArray64()
    alternative_masked_array_indexed = (
        alternative_masked_array.layout.to_IndexedOptionArray64()
    )
    assert ak.almost_equal(masked_array, alternative_masked_array_indexed)
    assert ak.almost_equal(masked_array_indexed, alternative_masked_array_indexed)


def test_record():
    array = ak.Array([{"x": 1}, {"x": 2}])

    assert not ak.almost_equal(array, [{"x": 1}, {"x": 3}])
    assert not ak.almost_equal(array, [{"x": 1}])
    assert not ak.almost_equal(array, [{"x": 1}, {"x": 2.0}])
    assert ak.almost_equal(array, [{"x": 1}, {"x": 2}])


def test_union():
    array = ak.Array([{"x": 1}, {"y": 2}])

    assert not ak.almost_equal(array, [{"x": 1}, {"y": 3}])
    assert not ak.almost_equal(array, [{"x": 1}])
    assert not ak.almost_equal(array, [{"x": 1}, {"y": 2.0}])
    assert ak.almost_equal(array, [{"x": 1}, {"y": 2}])


def test_empty():
    array = ak.Array([])

    assert not ak.almost_equal(array, [{"x": 1}, {"y": 3}])
    assert not ak.almost_equal(array, [1])
    assert not ak.almost_equal(array, ["hello"])
    assert not ak.almost_equal(array, [False, False])
    assert ak.almost_equal(array, ak.contents.EmptyArray())


def test_behavior():
    class CustomList(ak.Array): ...

    behavior = {"custom_list": CustomList}

    array = ak.with_parameter([[1, 2, 3]], "__list__", "custom_list", behavior=behavior)
    other_array = ak.with_parameter([[1, 2, 3]], "__list__", "custom_list")
    assert not ak.almost_equal(array, other_array)
    assert ak.almost_equal(array, other_array, check_parameters=False)

    another_array = ak.Array([[1, 2, 3]], behavior=behavior)
    assert not ak.almost_equal(array, another_array)
    assert not ak.almost_equal(other_array, another_array)


def test_empty_outer_ragged():
    array = ak.Array([[1]])[0:0]
    assert not ak.almost_equal(array, [])
    assert ak.almost_equal(array, array)


def test_numpy_array():
    left = np.arange(2 * 3 * 4, dtype=np.int64).reshape(4, 3, 2)
    right = np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4)
    assert not ak.almost_equal(left, right)
    assert ak.almost_equal(left, left)


def test_typetracer():
    array = ak.Array([[[1, 2, 3]], [[5, 4]]], backend="typetracer")
    with pytest.raises(NotImplementedError):
        ak.almost_equal(array, 2 * array)


def test_indexed():
    assert ak.almost_equal(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 2, 4, 8]),
            ak.contents.IndexedArray(
                ak.index.Index64([0, 1, 2, 3, 2, 1, 0, 5]),
                ak.contents.NumpyArray(np.arange(6, dtype=np.int64)),
            ),
        ),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 2, 4, 8]),
            ak.contents.NumpyArray(np.array([0, 1, 2, 3, 2, 1, 0, 5], dtype=np.int64)),
        ),
    )
