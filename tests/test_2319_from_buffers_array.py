# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_1d_1d_stride_trick():
    data = np.array([101], dtype=np.int32)
    array = np.lib.stride_tricks.as_strided(data, (40,), strides=(0,))

    container = {"node0-data": array}

    form = """
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "form_key": "node0"
        }
    """

    result = ak.from_buffers(form, array.size, container, highlevel=False)
    assert np.shares_memory(result.data, array)
    assert result.shape == (array.size,)


def test_1d_1d_different_dtypes_stride_trick():
    data = np.array([101], dtype=np.int64)
    array = np.lib.stride_tricks.as_strided(data, (40,), strides=(0,))

    container = {"node0-data": array}

    form = """
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "form_key": "node0"
        }
    """

    # We can't change the array view if the buffer is not contiguous
    with pytest.raises(ValueError, match="contiguous"):
        ak.from_buffers(form, array.size, container, highlevel=False)


def test_1d_1d_different_dtypes_contiguous():
    array = np.arange(40, dtype=np.int64)

    container = {"node0-data": array}

    form = """
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "form_key": "node0"
        }
    """
    result = ak.from_buffers(form, array.size, container, highlevel=False)
    assert np.shares_memory(array, result.data)


def test_2d_1d_fortran_ordered():
    array = np.arange(40 * 3, dtype=np.int32).reshape(40, 3).T

    container = {"node0-data": array}
    form = """
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "form_key": "node0"
        }
    """

    with pytest.raises(ValueError, match="cannot reshape array without copying"):
        ak.from_buffers(form, array.size, container, highlevel=False)


def test_2d_2d_fortran_ordered():
    array = np.arange(40 * 3, dtype=np.int32).reshape(40, 3).T

    container = {"node0-data": array}
    form = """
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "form_key": "node0",
            "inner_shape": [3]
        }
    """

    with pytest.raises(ValueError, match="cannot reshape array without copying"):
        ak.from_buffers(form, array.size, container, highlevel=False)


def test_2d_2d_stride_trick():
    data = np.array([101], dtype=np.int32)
    array = np.lib.stride_tricks.as_strided(data, (40, 3), strides=(0, 0))

    container = {"node0-data": array}
    form = """
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "form_key": "node0",
            "inner_shape": [3]
        }
    """

    result = ak.from_buffers(form, len(array), container, highlevel=False)
    assert np.shares_memory(result.data, array)

    # Read less than we have available
    result = ak.from_buffers(form, len(array) - 4, container, highlevel=False)
    assert np.shares_memory(result.data, array)


def test_2d_2d_different_stride_trick():
    data = np.array([101], dtype=np.int32)
    array = np.lib.stride_tricks.as_strided(data, (40, 3), strides=(0, 0))

    container = {"node0-data": array}
    form = """
        {
            "class": "NumpyArray",
            "primitive": "int32",
            "form_key": "node0",
            "inner_shape": [8]
        }
    """

    result = ak.from_buffers(form, array.size // 8, container, highlevel=False)
    assert np.shares_memory(result.data, array)

    # Read less than we have available
    result = ak.from_buffers(form, array.size // 8 - 4, container, highlevel=False)
    assert np.shares_memory(result.data, array)


def test_round_trip():
    data = np.arange(9 * 5 * 3).reshape((5, 9, 3))
    array = ak.from_numpy(data)

    result = ak.from_buffers(*ak.to_buffers(array), highlevel=False)
    assert np.shares_memory(result.data, data)
    assert ak.almost_equal(array, result)


def test_round_strided():
    data = np.array([101], dtype=np.int32)
    array = np.lib.stride_tricks.as_strided(data, (100,), strides=(0,))

    result = ak.from_buffers(*ak.to_buffers(array), highlevel=False)
    assert np.shares_memory(result.data, data)
    assert ak.almost_equal(array, result)
