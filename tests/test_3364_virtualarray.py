# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._backends.dispatch import backend_of_obj
from awkward._nplikes.dispatch import nplike_of_obj
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.virtual import VirtualArray, materialize_if_virtual


# Create fixtures for common test setup
@pytest.fixture
def numpy_like():
    return Numpy.instance()


@pytest.fixture
def simple_array_generator():
    return lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64)


@pytest.fixture
def virtual_array(numpy_like, simple_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=simple_array_generator,
    )


@pytest.fixture
def two_dim_array_generator():
    return lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)


@pytest.fixture
def two_dim_virtual_array(numpy_like, two_dim_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=two_dim_array_generator,
    )


@pytest.fixture
def scalar_array_generator():
    return lambda: np.array(42, dtype=np.int64)


@pytest.fixture
def scalar_virtual_array(numpy_like, scalar_array_generator):
    return VirtualArray(
        numpy_like, shape=(), dtype=np.dtype(np.int64), generator=scalar_array_generator
    )


@pytest.fixture
def float_array_generator():
    return lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)


@pytest.fixture
def float_virtual_array(numpy_like, float_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=float_array_generator,
    )


@pytest.fixture
def numpyarray():
    return ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64))


@pytest.fixture
def virtual_numpyarray(float_virtual_array):
    return ak.contents.NumpyArray(float_virtual_array)


@pytest.fixture
def offset_array_generator():
    return lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64)


@pytest.fixture
def virtual_offset_array(numpy_like, offset_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=offset_array_generator,
    )


@pytest.fixture
def listoffsetarray():
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    return ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )


@pytest.fixture
def virtual_listoffsetarray(numpy_like, virtual_offset_array, virtual_content_array):
    return ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offset_array),
        ak.contents.NumpyArray(virtual_content_array),
    )


@pytest.fixture
def starts_array_generator():
    return lambda: np.array([0, 2, 4, 7], dtype=np.int64)


@pytest.fixture
def virtual_starts_array(numpy_like, starts_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=starts_array_generator,
    )


@pytest.fixture
def stops_array_generator():
    return lambda: np.array([2, 4, 7, 10], dtype=np.int64)


@pytest.fixture
def virtual_stops_array(numpy_like, stops_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=stops_array_generator,
    )


@pytest.fixture
def content_array_generator():
    return lambda: np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )


@pytest.fixture
def virtual_content_array(numpy_like, content_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=content_array_generator,
    )


@pytest.fixture
def listarray():
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    return ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )


@pytest.fixture
def virtual_listarray(
    numpy_like, virtual_starts_array, virtual_stops_array, virtual_content_array
):
    return ak.contents.ListArray(
        ak.index.Index(virtual_starts_array),
        ak.index.Index(virtual_stops_array),
        ak.contents.NumpyArray(virtual_content_array),
    )


@pytest.fixture
def offsets_array_generator():
    return lambda: np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)


@pytest.fixture
def virtual_offsets_array(numpy_like, offsets_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=offsets_array_generator,
    )


@pytest.fixture
def x_content_array_generator():
    return lambda: np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )


@pytest.fixture
def virtual_x_content_array(numpy_like, x_content_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=x_content_array_generator,
    )


@pytest.fixture
def y_array_generator():
    return lambda: np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)


@pytest.fixture
def virtual_y_array(numpy_like, y_array_generator):
    return VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=y_array_generator,
    )


@pytest.fixture
def recordarray():
    # Create a regular RecordArray with ListOffsetArray and NumpyArray fields
    offsets = np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)
    x_content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    y_content = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )

    y_field = ak.contents.NumpyArray(y_content)

    return ak.contents.RecordArray([x_field, y_field], ["x", "y"])


@pytest.fixture
def virtual_recordarray(
    numpy_like, virtual_offsets_array, virtual_x_content_array, virtual_y_array
):
    # Create a RecordArray with virtual components
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets_array),
        ak.contents.NumpyArray(virtual_x_content_array),
    )

    y_field = ak.contents.NumpyArray(virtual_y_array)

    return ak.contents.RecordArray([x_field, y_field], ["x", "y"])


# Test initialization
def test_init_valid(numpy_like, simple_array_generator):
    va = VirtualArray(
        numpy_like, shape=(5,), dtype=np.int64, generator=simple_array_generator
    )
    assert va.shape == (5,)
    assert va.dtype == np.dtype(np.int64)
    assert not va.is_materialized


def test_init_invalid_shape():
    nplike = Numpy.instance()
    with pytest.raises(
        TypeError,
        match=r"Only shapes of integer dimensions or unknown_length are supported",
    ):
        VirtualArray(
            nplike,
            shape=("not_an_integer", 5),
            dtype=np.int64,
            generator=lambda: np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
        )


# Test properties
def test_dtype(virtual_array):
    assert virtual_array.dtype == np.dtype(np.int64)


def test_shape(virtual_array, two_dim_virtual_array):
    assert virtual_array.shape == (5,)
    assert two_dim_virtual_array.shape == (2, 3)


def test_ndim(virtual_array, two_dim_virtual_array, scalar_virtual_array):
    assert virtual_array.ndim == 1
    assert two_dim_virtual_array.ndim == 2
    assert scalar_virtual_array.ndim == 0


def test_size(virtual_array, two_dim_virtual_array, scalar_virtual_array):
    assert virtual_array.size == 5
    assert two_dim_virtual_array.size == 6
    assert scalar_virtual_array.size == 1


def test_nbytes_unmaterialized(virtual_array):
    assert virtual_array.nbytes == virtual_array.size * virtual_array.dtype.itemsize


def test_nbytes_materialized(virtual_array):
    virtual_array.materialize()
    assert virtual_array.nbytes == np.array([1, 2, 3, 4, 5], dtype=np.int64).nbytes


def test_strides(virtual_array):
    expected_strides = np.array([1, 2, 3, 4, 5], dtype=np.int64).strides
    assert virtual_array.strides == expected_strides


# Test materialization
def test_materialize(virtual_array):
    result = virtual_array.materialize()
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))
    assert virtual_array.is_materialized


def test_is_materialized(virtual_array):
    assert not virtual_array.is_materialized
    virtual_array.materialize()
    assert virtual_array.is_materialized


def test_materialize_shape_mismatch(numpy_like):
    # Generator returns array with different shape than declared
    with pytest.raises(
        ValueError,
        match=r"had shape \(5,\) before materialization while the materialized array has shape \(3,\)",
    ):
        va = VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.int64,
            generator=lambda: np.array([1, 2, 3], dtype=np.int64),
        )
        va.materialize()


def test_materialize_dtype_mismatch(numpy_like):
    # Generator returns array with different dtype than declared
    with pytest.raises(
        ValueError,
        match=r"had dtype int64 before materialization while the materialized array has dtype float64",
    ):
        va = VirtualArray(
            numpy_like,
            shape=(3,),
            dtype=np.int64,
            generator=lambda: np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        va.materialize()


# Test transpose
def test_T_unmaterialized(two_dim_virtual_array):
    transposed = two_dim_virtual_array.T
    assert isinstance(transposed, VirtualArray)
    assert transposed.shape == (3, 2)
    assert not transposed.is_materialized


def test_T_materialized(two_dim_virtual_array):
    two_dim_virtual_array.materialize()
    transposed = two_dim_virtual_array.T
    assert isinstance(transposed, np.ndarray)
    assert transposed.shape == (3, 2)


# Test view
def test_view_unmaterialized(virtual_array):
    view = virtual_array.view(np.float64)
    assert isinstance(view, VirtualArray)
    assert view.dtype == np.dtype(np.float64)
    assert not view.is_materialized


def test_view_materialized(virtual_array):
    virtual_array.materialize()
    view = virtual_array.view(np.float64)
    assert isinstance(view, np.ndarray)
    assert view.dtype == np.dtype(np.float64)


def test_view_invalid_size():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(3,),
        dtype=np.int8,
        generator=lambda: np.array([1, 2, 3], dtype=np.int8),
    )
    with pytest.raises(
        ValueError, match=r"new size of array with larger dtype must be a divisor"
    ):
        va.view(np.int32)


# Test generator property
def test_generator(virtual_array, simple_array_generator):
    assert virtual_array._generator is simple_array_generator


# Test nplike property
def test_nplike(virtual_array, numpy_like):
    assert virtual_array.nplike is numpy_like


# Test copy
def test_copy(virtual_array):
    copy = virtual_array.copy()
    assert isinstance(copy, VirtualArray)
    assert copy.shape == virtual_array.shape
    assert copy.dtype == virtual_array.dtype
    assert not copy.is_materialized  # Copy should not be materialized
    assert id(copy) != id(virtual_array)  # Different objects


# Test tolist
def test_tolist(virtual_array):
    assert virtual_array.tolist() == [1, 2, 3, 4, 5]


# Test tobytes
def test_tobytes(virtual_array):
    assert virtual_array.tobytes(order="C") == np.array(
        [1, 2, 3, 4, 5], dtype=np.int64
    ).tobytes(order="C")
    assert virtual_array.tobytes(order="F") == np.array(
        [1, 2, 3, 4, 5], dtype=np.int64
    ).tobytes(order="F")
    assert virtual_array.is_materialized


# Test ctypes
def test_ctypes(virtual_array):
    ctypes_data = virtual_array.ctypes
    assert ctypes_data is not None


# Test data
def test_data(virtual_array):
    data = virtual_array.data
    assert data is not None


# Test __repr__ and __str__
def test_repr(virtual_array):
    repr_str = repr(virtual_array)
    assert "VirtualArray" in repr_str
    assert "shape=(5,)" in repr_str


def test_str_scalar(scalar_virtual_array):
    assert str(scalar_virtual_array) == "??"


def test_str_array(virtual_array):
    str_val = str(virtual_array)
    assert "VirtualArray" in str_val


# Test __getitem__
def test_getitem_index(virtual_array):
    assert virtual_array[0] == 1
    assert virtual_array[4] == 5
    assert virtual_array[-1] == 5


def test_getitem_slice(virtual_array):
    sliced = virtual_array[1:4]
    assert isinstance(sliced, VirtualArray)
    assert sliced.shape == (3,)
    np.testing.assert_array_equal(sliced.materialize(), np.array([2, 3, 4]))


def test_getitem_slice_with_step(virtual_array):
    sliced = virtual_array[::2]
    assert isinstance(sliced, VirtualArray)
    assert sliced.shape == (3,)
    np.testing.assert_array_equal(sliced.materialize(), np.array([1, 3, 5]))


def test_getitem_slice_with_unknown_length():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(5,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
    )
    with pytest.raises(
        TypeError, match=r"does not support slicing with unknown_length"
    ):
        va[unknown_length:4]


# Test __setitem__
def test_setitem(virtual_array):
    virtual_array[2] = 10
    assert virtual_array[2] == 10
    np.testing.assert_array_equal(
        virtual_array.materialize(), np.array([1, 2, 10, 4, 5])
    )


# Test __bool__
def test_bool_scalar(scalar_virtual_array):
    assert bool(scalar_virtual_array) is True

    # Test with zero value
    nplike = Numpy.instance()
    va_zero = VirtualArray(
        nplike, shape=(), dtype=np.int64, generator=lambda: np.array(0, dtype=np.int64)
    )
    assert bool(va_zero) is False


def test_bool_array(virtual_array):
    with pytest.raises(
        ValueError,
        match=r"The truth value of an array with more than one element is ambiguous",
    ):
        bool(virtual_array)


# Test __int__
def test_int_scalar(scalar_virtual_array):
    assert int(scalar_virtual_array) == 42


def test_int_array(virtual_array):
    with pytest.raises(
        TypeError, match=r"Only scalar arrays can be converted to an int"
    ):
        int(virtual_array)


# Test __index__
def test_index_scalar(scalar_virtual_array):
    assert scalar_virtual_array.__index__() == 42


def test_index_array(virtual_array):
    with pytest.raises(TypeError, match=r"Only scalar arrays can be used as an index"):
        virtual_array.__index__()


# Test __len__
def test_len(virtual_array, two_dim_virtual_array):
    assert len(virtual_array) == 5
    assert len(two_dim_virtual_array) == 2


def test_len_scalar():
    # Scalar arrays don't have a length
    nplike = Numpy.instance()
    scalar_va = VirtualArray(
        nplike, shape=(), dtype=np.int64, generator=lambda: np.array(42, dtype=np.int64)
    )
    with pytest.raises(TypeError, match=r"len\(\) of unsized object"):
        len(scalar_va)


# Test __iter__
def test_iter(virtual_array):
    assert list(virtual_array) == [1, 2, 3, 4, 5]


# Test __dlpack__ and __dlpack_device__
@pytest.mark.skipif(
    tuple(map(int, np.__version__.split(".")[:2])) < (1, 23),
    reason="Test requires NumPy >= 1.23",
)
def test_dlpack_device(virtual_array):
    virtual_array.__dlpack_device__()


@pytest.mark.skipif(
    tuple(map(int, np.__version__.split(".")[:2])) < (1, 23),
    reason="Test requires NumPy >= 1.23",
)
def test_dlpack(virtual_array):
    virtual_array.__dlpack__()


# Test __array_ufunc__
def test_array_ufunc(virtual_array, monkeypatch):
    # Call a ufunc on the virtual array
    result = np.add(virtual_array, np.array([1, 2, 3, 4, 5]))
    assert virtual_array.is_materialized
    np.testing.assert_array_equal(result, np.array([2, 4, 6, 8, 10]))


# Test the helper function materialize_if_virtual
def test_materialize_if_virtual():
    from awkward._nplikes.virtual import materialize_if_virtual

    nplike = Numpy.instance()
    va1 = VirtualArray(
        nplike,
        shape=(3,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3], dtype=np.int64),
    )
    va2 = VirtualArray(
        nplike,
        shape=(2,),
        dtype=np.int64,
        generator=lambda: np.array([4, 5], dtype=np.int64),
    )
    regular_array = np.array([6, 7, 8])

    result = materialize_if_virtual(va1, regular_array, va2)

    assert len(result) == 3
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[2], np.ndarray)
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(result[1], np.array([6, 7, 8]))
    np.testing.assert_array_equal(result[2], np.array([4, 5]))


# Tests for float virtual array
def test_float_array_init(numpy_like, float_array_generator):
    va = VirtualArray(
        numpy_like, shape=(5,), dtype=np.float64, generator=float_array_generator
    )
    assert va.shape == (5,)
    assert va.dtype == np.dtype(np.float64)
    assert not va.is_materialized


def test_float_array_materialize(float_virtual_array):
    result = float_virtual_array.materialize()
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert float_virtual_array.is_materialized


def test_float_array_slicing(numpy_like, float_array_generator):
    # Test basic slice
    float_virtual_array1 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=float_array_generator,
    )
    sliced = float_virtual_array1[1:4]
    assert isinstance(sliced, VirtualArray)
    assert sliced.shape == (3,)
    np.testing.assert_array_almost_equal(
        sliced.materialize(), np.array([2.2, 3.3, 4.4])
    )

    # Test step slice
    float_virtual_array2 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=float_array_generator,
    )
    sliced_step = float_virtual_array2[::2]
    assert isinstance(sliced_step, VirtualArray)
    assert sliced_step.shape == (3,)
    np.testing.assert_array_almost_equal(
        sliced_step.materialize(), np.array([1.1, 3.3, 5.5])
    )

    # Test negative step
    float_virtual_array3 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=float_array_generator,
    )
    sliced_neg = float_virtual_array3[::-1]
    assert isinstance(sliced_neg, VirtualArray)
    assert sliced_neg.shape == (5,)
    np.testing.assert_array_almost_equal(
        sliced_neg.materialize(), np.array([5.5, 4.4, 3.3, 2.2, 1.1])
    )

    # Test complex slice
    float_virtual_array4 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=float_array_generator,
    )
    sliced_complex = float_virtual_array4[4:1:-2]
    assert isinstance(sliced_complex, VirtualArray)
    assert sliced_complex.shape == (2,)
    np.testing.assert_array_almost_equal(
        sliced_complex.materialize(), np.array([5.5, 3.3])
    )


def test_float_array_operations(float_virtual_array):
    # Test arithmetic operations
    result = float_virtual_array + 1.0
    np.testing.assert_array_almost_equal(result, np.array([2.1, 3.2, 4.3, 5.4, 6.5]))

    # Test multiplication
    result = float_virtual_array * 2.0
    np.testing.assert_array_almost_equal(result, np.array([2.2, 4.4, 6.6, 8.8, 11.0]))

    # Test division
    result = float_virtual_array / 2.0
    np.testing.assert_array_almost_equal(result, np.array([0.55, 1.1, 1.65, 2.2, 2.75]))

    # Test mixed operation with integer array
    int_array = np.array([1, 2, 3, 4, 5])
    result = float_virtual_array + int_array
    np.testing.assert_array_almost_equal(result, np.array([2.1, 4.2, 6.3, 8.4, 10.5]))


def test_float_array_view(float_virtual_array):
    # Test view as different float type
    view = float_virtual_array.view(np.float32)
    assert isinstance(view, VirtualArray)
    assert view.dtype == np.dtype(np.float32)

    # Test materialization of view
    materialized = view.materialize()
    assert materialized.dtype == np.dtype(np.float32)


def test_float_to_int_comparison(float_virtual_array, virtual_array):
    # Compare float and int arrays
    float_data = float_virtual_array.materialize()
    int_data = virtual_array.materialize()

    # Test basic properties
    assert float_virtual_array.shape == virtual_array.shape
    assert float_virtual_array.ndim == virtual_array.ndim
    assert float_virtual_array.size == virtual_array.size

    # Test conversion between types
    int_view = float_virtual_array.view(np.int64)
    assert int_view.dtype == np.dtype(np.int64)

    # Test operations between float and int arrays
    result = float_virtual_array + virtual_array
    expected = float_data + int_data
    np.testing.assert_array_almost_equal(result, expected)


# Test rounding operations specific to float arrays
def test_float_array_rounding():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(5,),
        dtype=np.float64,
        generator=lambda: np.array([1.1, 2.5, 3.7, 4.2, 5.9], dtype=np.float64),
    )

    # Test floor
    result = np.floor(va)
    np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    # Test ceil
    result = np.ceil(va)
    np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0, 4.0, 5.0, 6.0]))

    # Test round
    result = np.round(va)
    np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 4.0, 4.0, 6.0]))


# Test NaN handling
def test_float_array_nan():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(5,),
        dtype=np.float64,
        generator=lambda: np.array([1.1, np.nan, 3.3, np.nan, 5.5], dtype=np.float64),
    )

    # Test isnan
    result = np.isnan(va)
    np.testing.assert_array_equal(result, np.array([False, True, False, True, False]))

    # Test nanmean
    result = np.nanmean(va)
    np.testing.assert_almost_equal(result, (1.1 + 3.3 + 5.5) / 3)


# Multidimensional slicing tests
def test_multidim_slicing(two_dim_virtual_array):
    # Test slicing on first dimension
    sliced = two_dim_virtual_array[0]
    assert isinstance(sliced, np.ndarray)  # Should be materialized
    np.testing.assert_array_equal(sliced, np.array([1, 2, 3]))

    # Fresh array for next test to avoid materialization effects
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(2, 3),
        dtype=np.int64,
        generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    )

    # Test advanced indexing
    sliced = va[:, 1]
    # This should materialize because it's not a simple first-dimension slice
    assert isinstance(sliced, np.ndarray)
    np.testing.assert_array_equal(sliced, np.array([2, 5]))


# Test empty array handling
def test_empty_array():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(0,),
        dtype=np.int64,
        generator=lambda: np.array([], dtype=np.int64),
    )

    assert va.shape == (0,)
    assert va.size == 0
    assert len(va) == 0
    materialized = va.materialize()
    assert len(materialized) == 0


# Test with structured dtypes
def test_structured_dtype():
    dtype = np.dtype([("name", "U10"), ("age", "i4"), ("weight", "f8")])
    data = np.array(
        [("Alice", 25, 55.0), ("Bob", 30, 70.5), ("Charlie", 35, 65.2)], dtype=dtype
    )

    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(3,),
        dtype=dtype,
        generator=lambda: data,
    )

    assert va.dtype == dtype
    materialized = va.materialize()
    assert materialized["name"][1] == "Bob"
    assert materialized["age"][2] == 35
    assert materialized["weight"][0] == 55.0


# Test with large arrays to check memory efficiency
def test_large_array_memory():
    # Create a large array that would consume significant memory
    nplike = Numpy.instance()

    # Define a generator that would create a large array
    def large_array_generator():
        return np.ones((1000, 1000), dtype=np.float64)

    va = VirtualArray(
        nplike,
        shape=(1000, 1000),
        dtype=np.float64,
        generator=large_array_generator,
    )

    assert va.nbytes == 1000 * 1000 * 8  # 8 bytes per float64 element

    # Access just one element to check if full materialization happens
    element = va[0, 0]
    assert element == 1.0

    # Now the array should be materialized
    assert va.nbytes == 1000 * 1000 * 8  # Should still be the same
    assert va.is_materialized


# Test error propagation from generator
def test_generator_error():
    nplike = Numpy.instance()

    def failing_generator():
        raise ValueError("Generator failure test")

    va = VirtualArray(
        nplike,
        shape=(5,),
        dtype=np.int64,
        generator=failing_generator,
    )

    with pytest.raises(ValueError, match="Generator failure test"):
        va.materialize()


# Test nested VirtualArrays (generator returns another VirtualArray)
def test_nested_virtual_arrays():
    nplike = Numpy.instance()

    # Inner virtual array
    inner_va = VirtualArray(
        nplike,
        shape=(3,),
        dtype=np.int64,
        generator=lambda: np.array([10, 20, 30], np.int64),
    )

    # Outer virtual array, generator returns inner virtual array
    outer_va = VirtualArray(
        nplike,
        shape=(3,),
        dtype=np.int64,
        generator=lambda: inner_va,
    )

    # Should materialize both
    result = outer_va.materialize()
    np.testing.assert_array_equal(result, np.array([10, 20, 30]))
    assert inner_va.is_materialized
    assert outer_va.is_materialized


# Test with complex numbers
def test_complex_numbers():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(3,),
        dtype=np.complex128,
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    assert va.dtype == np.dtype(np.complex128)
    materialized = va.materialize()
    np.testing.assert_array_equal(materialized, np.array([1 + 2j, 3 + 4j, 5 + 6j]))

    # Test complex operations
    result = va * (2 + 1j)
    expected = np.array([1 + 2j, 3 + 4j, 5 + 6j]) * (2 + 1j)
    np.testing.assert_array_almost_equal(result, expected)


# Test slice with 0 step raises error
def test_slice_zero_step():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(5,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
    )

    with pytest.raises(ValueError):
        va[::0]  # Step can't be zero


def test_slice_length_calculation():
    nplike = Numpy.instance()
    test_cases = [
        # (slice, expected_length, array_length)
        (slice(None), 5, 5),  # [:] -> 5
        (slice(1, 4), 3, 5),  # [1:4] -> 3
        (slice(None, None, 2), 3, 5),  # [::2] -> 3
        (slice(None, None, -1), 5, 5),  # [::-1] -> 5
        (slice(4, 1, -1), 3, 5),  # [4:1:-1] -> 3
        (slice(1, 10), 4, 5),  # [1:10] -> 4 (out of bounds)
        (slice(-10, 10), 5, 5),  # [-10:10] -> 5 (both out of bounds)
        (slice(10, 20), 0, 5),  # [10:20] -> 0 (start beyond length)
        (slice(None, None, 3), 2, 5),  # [::3] -> 2
    ]

    for slice_obj, expected_length, array_length in test_cases:
        # Create a closure that captures the current value of array_length
        def create_generator(length):
            return lambda: np.ones(length, dtype=np.int64)

        va = VirtualArray(
            nplike,
            shape=(array_length,),
            dtype=np.int64,
            generator=create_generator(array_length),
        )

        sliced = va[slice_obj]
        assert isinstance(sliced, VirtualArray)
        assert sliced.shape[0] == expected_length, f"Failed for slice {slice_obj}"


# Test nplike of obj
def test_nplike_of_obj(virtual_array, float_virtual_array, numpy_like):
    assert nplike_of_obj(virtual_array) is numpy_like
    assert nplike_of_obj(float_virtual_array) is numpy_like


# Test backend of obj
def test_backend_of_obj(virtual_array, float_virtual_array):
    assert backend_of_obj(virtual_array).name == "cpu"
    assert backend_of_obj(float_virtual_array).name == "cpu"


# Test array creation methods with VirtualArray
def test_asarray_virtual_array_unmaterialized(numpy_like, virtual_array):
    # Test with unmaterialized VirtualArray
    result = numpy_like.asarray(virtual_array)
    assert result is virtual_array  # Should return the same object
    assert not virtual_array.is_materialized


def test_asarray_virtual_array_materialized(numpy_like, virtual_array):
    # Test with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.asarray(virtual_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_asarray_virtual_array_with_dtype(numpy_like, virtual_array):
    # Test with dtype parameter
    out = numpy_like.asarray(virtual_array, dtype=np.float64)
    assert isinstance(out, VirtualArray)
    assert out.dtype == np.dtype(np.float64)
    assert not out.is_materialized
    assert out.materialize().dtype == np.dtype(np.float64)


def test_asarray_virtual_array_with_copy(numpy_like, virtual_array):
    # Test with copy parameter
    virtual_array.materialize()
    with pytest.raises(
        ValueError,
        match="asarray was called with copy=False for an array of a different dtype",
    ):
        # Should raise because we're trying to change the dtype without copying
        numpy_like.asarray(virtual_array, dtype=np.float64, copy=False)


def test_ascontiguousarray_unmaterialized(numpy_like, virtual_array):
    # Test with unmaterialized VirtualArray
    result = numpy_like.ascontiguousarray(virtual_array)
    assert isinstance(result, VirtualArray)
    assert not result.is_materialized
    assert result.shape == virtual_array.shape
    assert result.dtype == virtual_array.dtype


def test_ascontiguousarray_materialized(numpy_like, virtual_array):
    # Test with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.ascontiguousarray(virtual_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_frombuffer_with_virtual_array(numpy_like, virtual_array):
    # Test frombuffer with VirtualArray (should raise TypeError)
    with pytest.raises(
        TypeError, match="virtual arrays are not supported in `frombuffer`"
    ):
        numpy_like.frombuffer(virtual_array)


# Test array creation methods using materialization info
def test_zeros_like_unmaterialized(numpy_like, virtual_array):
    # Test zeros_like with unmaterialized VirtualArray
    result = numpy_like.zeros_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.zeros(5, dtype=np.int64))
    assert not virtual_array.is_materialized


def test_zeros_like_materialized(numpy_like, virtual_array):
    # Test zeros_like with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.zeros_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.zeros(5, dtype=np.int64))


def test_ones_like_unmaterialized(numpy_like, virtual_array):
    # Test ones_like with unmaterialized VirtualArray
    result = numpy_like.ones_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.ones(5, dtype=np.int64))
    assert not virtual_array.is_materialized


def test_ones_like_materialized(numpy_like, virtual_array):
    # Test ones_like with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.ones_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.ones(5, dtype=np.int64))


def test_full_like_unmaterialized(numpy_like, virtual_array):
    # Test full_like with unmaterialized VirtualArray
    result = numpy_like.full_like(virtual_array, 7)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.full(5, 7, dtype=np.int64))
    assert not virtual_array.is_materialized


def test_full_like_materialized(numpy_like, virtual_array):
    # Test full_like with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.full_like(virtual_array, 7)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.full(5, 7, dtype=np.int64))


# Test arange and meshgrid with VirtualArray parameters
def test_arange_with_virtual_array_start(numpy_like, scalar_virtual_array):
    # Test arange with VirtualArray parameter
    arange = numpy_like.arange(scalar_virtual_array, 10)
    assert scalar_virtual_array.is_materialized
    np.testing.assert_array_equal(arange, np.arange(42, 10))


def test_meshgrid_with_virtual_array(numpy_like, virtual_array):
    # Test meshgrid with VirtualArray parameter
    virtual_array.materialize()
    result = numpy_like.meshgrid(virtual_array)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3, 4, 5]))


# Test testing functions with VirtualArray
def test_array_equal_with_virtual_arrays(numpy_like, virtual_array):
    # Create two identical VirtualArrays
    va1 = virtual_array
    va2 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
    )

    # Test array_equal
    result = numpy_like.array_equal(va1, va2)
    assert result is True

    # Test with a different VirtualArray
    va3 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array(
            [1, 2, 3, 4, 6], dtype=np.int64
        ),  # Different last value
    )
    result = numpy_like.array_equal(va1, va3)
    assert result is False


def test_array_equal_with_equal_nan(numpy_like):
    # Test array_equal with equal_nan=True
    va1 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.0, np.nan, 3.0], dtype=np.float64),
    )
    va2 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.0, np.nan, 3.0], dtype=np.float64),
    )

    # Should be False by default (NaN != NaN)
    result = numpy_like.array_equal(va1, va2)
    assert bool(result) is False

    # Should be True with equal_nan=True
    result = numpy_like.array_equal(va1, va2, equal_nan=True)
    assert bool(result) is True


def test_searchsorted_with_virtual_arrays(numpy_like, virtual_array):
    # Test searchsorted with VirtualArray
    values = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3, 6], dtype=np.int64),
    )

    result = numpy_like.searchsorted(virtual_array, values)
    np.testing.assert_array_equal(
        result, np.array([0, 2, 5])
    )  # Indices where values would be inserted


# Test ufunc application with VirtualArray
def test_apply_ufunc_with_virtual_arrays(numpy_like, virtual_array):
    # Test apply_ufunc with add operation
    result = numpy_like.apply_ufunc(np.add, "__call__", [virtual_array, 10])
    np.testing.assert_array_equal(result, np.array([11, 12, 13, 14, 15]))

    # Test apply_ufunc with multiple VirtualArrays
    va2 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
    )

    result = numpy_like.apply_ufunc(np.multiply, "__call__", [virtual_array, va2])
    np.testing.assert_array_equal(result, np.array([10, 40, 90, 160, 250]))


# Test manipulation functions with VirtualArray
def test_broadcast_arrays_with_virtual_arrays(numpy_like, virtual_array):
    # Test broadcast_arrays with VirtualArrays
    va2 = VirtualArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([10], dtype=np.int64),
    )

    result = numpy_like.broadcast_arrays(virtual_array, va2)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(result[1], np.array([10, 10, 10, 10, 10]))


def test_reshape_unmaterialized(numpy_like, virtual_array):
    # Test reshape with unmaterialized VirtualArray
    result = numpy_like.reshape(virtual_array, (5, 1))
    assert isinstance(result, VirtualArray)
    assert not result.is_materialized
    assert result.shape == (5, 1)

    # Test reshape with -1 dimension
    result = numpy_like.reshape(virtual_array, (-1, 1))
    assert isinstance(result, VirtualArray)
    assert not result.is_materialized
    assert result.shape == (5, 1)


def test_reshape_materialized(numpy_like, virtual_array):
    # Test reshape with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.reshape(virtual_array, (5, 1))
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 1)

    # Test reshape with copy=True
    result = numpy_like.reshape(virtual_array, (5, 1), copy=True)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 1)

    # Test reshape with copy=False
    result = numpy_like.reshape(virtual_array, (5, 1), copy=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 1)


def test_derive_slice_for_length(numpy_like):
    # Test derive_slice_for_length method
    slice_obj = slice(1, 4, 1)
    start, stop, step, slice_length = numpy_like.derive_slice_for_length(slice_obj, 5)
    assert start == 1
    assert stop == 4
    assert step == 1
    assert slice_length == 3

    # Test with negative step
    slice_obj = slice(4, 1, -1)
    start, stop, step, slice_length = numpy_like.derive_slice_for_length(slice_obj, 5)
    assert start == 4
    assert stop == 1
    assert step == -1
    assert slice_length == 3

    # Test with None values
    slice_obj = slice(None, None, 2)
    start, stop, step, slice_length = numpy_like.derive_slice_for_length(slice_obj, 5)
    assert start == 0
    assert stop == 5
    assert step == 2
    assert slice_length == 3


def test_nonzero_with_virtual_array(numpy_like, virtual_array):
    # Test nonzero with VirtualArray
    result = numpy_like.nonzero(virtual_array)
    assert len(result) == 1
    np.testing.assert_array_equal(
        result[0], np.array([0, 1, 2, 3, 4])
    )  # All values are non-zero


def test_where_with_virtual_arrays(numpy_like, virtual_array):
    # Test where with VirtualArrays
    condition = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, False, True, False, True], dtype=np.bool_),
    )

    x1 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
    )

    result = numpy_like.where(condition, virtual_array, x1)
    np.testing.assert_array_equal(result, np.array([1, 20, 3, 40, 5]))


def test_unique_values_with_virtual_array(numpy_like):
    # Test unique_values with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(7,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 2, 3, 3, 3, 1], dtype=np.int64),
    )

    result = numpy_like.unique_values(va)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_unique_all_with_virtual_array(numpy_like):
    # Test unique_all with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(7,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 2, 3, 3, 3, 1], dtype=np.int64),
    )

    result = numpy_like.unique_all(va)
    np.testing.assert_array_equal(result.values, np.array([1, 2, 3]))
    np.testing.assert_array_equal(result.counts, np.array([2, 2, 3]))
    # Check inverse indices have original shape
    assert result.inverse_indices.shape == (7,)


def test_sort_with_virtual_array(numpy_like):
    # Test sort with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
    )

    # Sort ascending
    result = numpy_like.sort(va)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))

    # Sort descending
    result = numpy_like.sort(va, descending=True)
    np.testing.assert_array_equal(result, np.array([5, 4, 3, 2, 1]))

    # Test with 2D array and axis
    va2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
    )

    result = numpy_like.sort(va2d, axis=1)
    np.testing.assert_array_equal(result, np.array([[1, 2, 3], [4, 5, 6]]))


def test_concat_with_virtual_arrays(numpy_like, virtual_array):
    # Test concat with VirtualArrays
    va2 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([6, 7, 8], dtype=np.int64),
    )

    result = numpy_like.concat([virtual_array, va2])
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    # Test with axis parameter
    va2d1 = VirtualArray(
        numpy_like,
        shape=(2, 2),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[1, 2], [3, 4]], dtype=np.int64),
    )

    va2d2 = VirtualArray(
        numpy_like,
        shape=(2, 2),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[5, 6], [7, 8]], dtype=np.int64),
    )

    result = numpy_like.concat([va2d1, va2d2], axis=1)
    np.testing.assert_array_equal(result, np.array([[1, 2, 5, 6], [3, 4, 7, 8]]))


def test_repeat_with_virtual_array(numpy_like, virtual_array):
    # Test repeat with VirtualArray
    result = numpy_like.repeat(virtual_array, 2)
    np.testing.assert_array_equal(result, np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))

    # Test with axis parameter
    va2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    )

    result = numpy_like.repeat(va2d, 2, axis=0)
    np.testing.assert_array_equal(
        result, np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])
    )


def test_stack_with_virtual_arrays(numpy_like, virtual_array):
    # Test stack with VirtualArrays
    va2 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([6, 7, 8, 9, 10], dtype=np.int64),
    )

    result = numpy_like.stack([virtual_array, va2])
    np.testing.assert_array_equal(result, np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    # Test with axis parameter
    result = numpy_like.stack([virtual_array, va2], axis=1)
    np.testing.assert_array_equal(
        result, np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
    )


def test_packbits_with_virtual_array(numpy_like):
    # Test packbits with VirtualArray of booleans
    va = VirtualArray(
        numpy_like,
        shape=(8,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
    )

    result = numpy_like.packbits(va)
    np.testing.assert_array_equal(
        result, np.array([170], dtype=np.uint8)
    )  # 10101010 in binary = 170


def test_unpackbits_with_virtual_array(numpy_like):
    # Test unpackbits with VirtualArray of uint8
    va = VirtualArray(
        numpy_like,
        shape=(1,),
        dtype=np.dtype(np.uint8),
        generator=lambda: np.array([170], dtype=np.uint8),
    )

    result = numpy_like.unpackbits(va)
    np.testing.assert_array_equal(
        result, np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    )


def test_broadcast_to_with_virtual_array(numpy_like, virtual_array):
    # Test broadcast_to with VirtualArray
    result = numpy_like.broadcast_to(virtual_array, (3, 5))
    assert result.shape == (3, 5)

    # Check all rows are the same
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(result[1], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(result[2], np.array([1, 2, 3, 4, 5]))


def test_strides_with_virtual_array(numpy_like, virtual_array):
    # Test strides with VirtualArray
    # First test without materializing
    assert numpy_like.strides(virtual_array) == (8,)  # 8 bytes per int64

    # Then test after materializing
    virtual_array.materialize()
    assert numpy_like.strides(virtual_array) == (8,)


# Test addition and logical operations
def test_add_with_virtual_arrays(numpy_like, virtual_array):
    # Test add with VirtualArrays
    va2 = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
    )

    result = numpy_like.add(virtual_array, va2)
    np.testing.assert_array_equal(result, np.array([11, 22, 33, 44, 55]))


def test_logical_or_with_virtual_arrays(numpy_like):
    # Test logical_or with VirtualArrays
    va1 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
    )

    va2 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, True, False, False], dtype=np.bool_),
    )

    result = numpy_like.logical_or(va1, va2)
    np.testing.assert_array_equal(result, np.array([True, True, True, False]))


def test_logical_and_with_virtual_arrays(numpy_like):
    # Test logical_and with VirtualArrays
    va1 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
    )

    va2 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, True, False, False], dtype=np.bool_),
    )

    result = numpy_like.logical_and(va1, va2)
    np.testing.assert_array_equal(result, np.array([True, False, False, False]))


def test_logical_not_with_virtual_array(numpy_like):
    # Test logical_not with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
    )

    result = numpy_like.logical_not(va)
    np.testing.assert_array_equal(result, np.array([False, True, False, True]))


# Test mathematical operations
def test_sqrt_with_virtual_array(numpy_like):
    # Test sqrt with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float64),
    )

    result = numpy_like.sqrt(va)
    np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0, 4.0, 5.0]))


def test_exp_with_virtual_array(numpy_like):
    # Test exp with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.0, 1.0, 2.0], dtype=np.float64),
    )

    result = numpy_like.exp(va)
    np.testing.assert_array_almost_equal(result, np.array([1.0, np.e, np.e**2]))


def test_divide_with_virtual_arrays(numpy_like):
    # Test divide with VirtualArrays
    va1 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
    )

    va2 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float64),
    )

    result = numpy_like.divide(va1, va2)
    np.testing.assert_array_almost_equal(result, np.array([5.0, 5.0, 6.0, 5.0]))


# Test special operations
def test_nan_to_num_with_virtual_array(numpy_like):
    # Test nan_to_num with VirtualArray containing NaN and infinity
    va = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float64),
    )

    result = numpy_like.nan_to_num(va)
    # NaN becomes 0.0, inf becomes large finite number, -inf becomes large negative number
    assert result[0] == 1.0
    assert result[1] == 0.0
    assert result[2] > 1e300  # Large positive number
    assert result[3] < -1e300  # Large negative number


def test_isclose_with_virtual_arrays(numpy_like):
    # Test isclose with VirtualArrays
    va1 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64),
    )

    va2 = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.0001, 2.0, 3.1, np.nan], dtype=np.float64),
    )

    # Default tolerance
    result = numpy_like.isclose(va1, va2)
    np.testing.assert_array_equal(result, np.array([False, True, False, False]))

    # Custom tolerance
    result = numpy_like.isclose(va1, va2, rtol=0.05)
    np.testing.assert_array_equal(result, np.array([True, True, True, False]))

    # Equal_nan parameter
    result = numpy_like.isclose(va1, va2, equal_nan=True)
    np.testing.assert_array_equal(result, np.array([False, True, False, True]))


def test_isnan_with_virtual_array(numpy_like):
    # Test isnan with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float64),
    )

    result = numpy_like.isnan(va)
    np.testing.assert_array_equal(result, np.array([False, True, False, True]))


# Test reduction operations
def test_all_with_virtual_array(numpy_like):
    # Test all with VirtualArray
    va_all_true = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, True, True, True], dtype=np.bool_),
    )

    va_mixed = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([True, False, True, True], dtype=np.bool_),
    )

    # Test all(all True)
    result = numpy_like.all(va_all_true)
    assert result

    # Test all(mixed)
    result = numpy_like.all(va_mixed)
    assert not result

    # Test with axis parameter
    va_2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array(
            [[True, True, False], [True, True, True]], dtype=np.bool_
        ),
    )

    result = numpy_like.all(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([False, True]))

    # Test with keepdims
    result = numpy_like.all(va_2d, axis=1, keepdims=True)
    np.testing.assert_array_equal(result, np.array([[False], [True]]))


def test_any_with_virtual_array(numpy_like):
    # Test any with VirtualArray
    va_all_false = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([False, False, False, False], dtype=np.bool_),
    )

    va_mixed = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array([False, False, True, False], dtype=np.bool_),
    )

    # Test any(all False)
    result = numpy_like.any(va_all_false)
    assert not result

    # Test any(mixed)
    result = numpy_like.any(va_mixed)
    assert result

    # Test with axis parameter
    va_2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.bool_),
        generator=lambda: np.array(
            [[False, False, False], [False, True, False]], dtype=np.bool_
        ),
    )

    result = numpy_like.any(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([False, True]))


def test_min_with_virtual_array(numpy_like):
    # Test min with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
    )

    # Test overall min
    result = numpy_like.min(va)
    assert result == 1

    # Test with 2D array and axis
    va_2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
    )

    result = numpy_like.min(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([1, 4]))


def test_max_with_virtual_array(numpy_like):
    # Test max with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
    )

    # Test overall max
    result = numpy_like.max(va)
    assert result == 5

    # Test with 2D array and axis
    va_2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
    )

    result = numpy_like.max(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([3, 6]))


def test_count_nonzero_with_virtual_array(numpy_like):
    # Test count_nonzero with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 0, 3, 0, 5, 0], dtype=np.int64),
    )

    # Test overall count
    result = numpy_like.count_nonzero(va)
    assert result == 3

    # Test with 2D array and axis
    va_2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[1, 0, 2], [0, 4, 0]], dtype=np.int64),
    )

    result = numpy_like.count_nonzero(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([2, 1]))


def test_cumsum_with_virtual_array(numpy_like):
    # Test cumsum with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
    )

    # Test cumsum
    result = numpy_like.cumsum(va)
    np.testing.assert_array_equal(result, np.array([1, 3, 6, 10, 15]))

    # Test with 2D array and axis
    va_2d = VirtualArray(
        numpy_like,
        shape=(2, 3),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    )

    result = numpy_like.cumsum(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([[1, 3, 6], [4, 9, 15]]))


def test_real_imag_with_complex_virtual_array(numpy_like):
    # Test real and imag with complex VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    # Test real
    result = numpy_like.real(va)
    np.testing.assert_array_equal(result, np.array([1.0, 3.0, 5.0]))

    # Test imag
    result = numpy_like.imag(va)
    np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))


def test_angle_with_complex_virtual_array(numpy_like):
    # Test angle with complex VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array(
            [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
        ),
    )

    # Test angle in radians
    result = numpy_like.angle(va)
    np.testing.assert_array_almost_equal(
        result, np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    )

    # Test angle in degrees
    result = numpy_like.angle(va, deg=True)
    np.testing.assert_array_almost_equal(result, np.array([0, 90, 180, -90]))


def test_round_with_virtual_array(numpy_like):
    # Test round with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
    )

    # Test round to nearest integer
    result = numpy_like.round(va)
    np.testing.assert_array_equal(result, np.array([1.0, 3.0, 3.0, 5.0]))

    # Test round to 1 decimal place
    result = numpy_like.round(va, decimals=1)
    np.testing.assert_array_equal(result, np.array([1.2, 2.6, 3.5, 4.5]))

    # Test round to 2 decimal places
    result = numpy_like.round(va, decimals=2)
    np.testing.assert_array_equal(result, np.array([1.23, 2.57, 3.50, 4.50]))


def test_array_str_with_virtual_array_unmaterialized(numpy_like, virtual_array):
    # Test array_str with unmaterialized VirtualArray
    result = numpy_like.array_str(virtual_array)
    assert result == "[## ... ##]"


def test_array_str_with_virtual_array_materialized(numpy_like, virtual_array):
    # Test array_str with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.array_str(virtual_array)
    assert "[1 2 3 4 5]" in result


def test_astype_with_virtual_array(numpy_like, virtual_array):
    # Test astype with VirtualArray
    result = numpy_like.astype(virtual_array, np.float64)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert result.dtype == np.dtype(np.float64)

    # Test with copy=False
    result = numpy_like.astype(virtual_array, np.int64, copy=False)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))
    assert result.dtype == np.dtype(np.int64)


def test_can_cast_with_virtual_array_dtype(numpy_like, virtual_array):
    # Test can_cast with VirtualArray's dtype
    # int64 can be cast to float64 with same_kind casting
    assert numpy_like.can_cast(virtual_array.dtype, np.float64) is True

    # int64 can be cast to complex64 with same_kind casting
    assert numpy_like.can_cast(virtual_array.dtype, np.complex64) is True


# Test various combinations and edge cases
def test_materialize_if_virtual_function(numpy_like):
    # Test the materialize_if_virtual utility function directly

    # Create a mix of VirtualArrays and regular arrays
    va1 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 3], dtype=np.int64),
    )

    va2 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([4, 5, 6], dtype=np.int64),
    )

    regular_array = np.array([7, 8, 9])

    # Materialize none of them
    results = materialize_if_virtual(va1, va2, regular_array)
    assert len(results) == 3
    np.testing.assert_array_equal(results[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(results[1], np.array([4, 5, 6]))
    np.testing.assert_array_equal(results[2], np.array([7, 8, 9]))

    # Check that the VirtualArrays were materialized
    assert va1.is_materialized
    assert va2.is_materialized

    # Test with already materialized VirtualArrays
    va3 = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([10, 11], dtype=np.int64),
    )
    va3.materialize()  # Pre-materialize

    results = materialize_if_virtual(va3, regular_array)
    assert len(results) == 2
    np.testing.assert_array_equal(results[0], np.array([10, 11]))
    np.testing.assert_array_equal(results[1], np.array([7, 8, 9]))


def test_operations_with_multiple_virtual_arrays(numpy_like):
    # Test a complex operation involving multiple VirtualArrays
    va1 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    va2 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )

    va3 = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([7.0, 8.0, 9.0], dtype=np.float64),
    )

    # Expression: (va1 + va2) * va3
    # Should materialize all VirtualArrays
    result = numpy_like.add(va1, va2) * va3
    np.testing.assert_array_equal(
        result,
        np.array([35.0, 56.0, 81.0]),  # (1+4)*7, (2+5)*8, (3+6)*9
    )

    # Check that all VirtualArrays were materialized
    assert va1.is_materialized
    assert va2.is_materialized
    assert va3.is_materialized


def test_is_own_array_with_virtual_array(numpy_like):
    # Test is_own_array method with VirtualArray
    va = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 3], dtype=np.int64),
    )

    # Before materialization
    result = numpy_like.is_own_array(va)
    assert result  # Should be True because VirtualArray.nplike.ndarray is numpy.ndarray

    # After materialization
    va.materialize()
    result = numpy_like.is_own_array(va)
    assert result


def test_virtual_array_with_structured_dtype(numpy_like):
    # Test VirtualArray with structured dtype
    dtype = np.dtype([("name", "U10"), ("age", "i4"), ("weight", "f8")])

    va = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=dtype,
        generator=lambda: np.array(
            [("Alice", 25, 55.0), ("Bob", 30, 70.5)], dtype=dtype
        ),
    )

    # Test properties
    assert va.dtype == dtype
    assert va.shape == (2,)

    # Test materialization
    materialized = va.materialize()
    assert materialized["name"][0] == "Alice"
    assert materialized["age"][1] == 30
    assert materialized["weight"][1] == 70.5


def test_virtual_array_with_empty_array(numpy_like):
    # Test VirtualArray with empty array
    va = VirtualArray(
        numpy_like,
        shape=(0,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([], dtype=np.int64),
    )

    # Test properties
    assert va.shape == (0,)
    assert va.size == 0

    # Test materialization
    materialized = va.materialize()
    assert len(materialized) == 0


def test_chained_operations_materialization(numpy_like):
    # Test that chained operations correctly materialize VirtualArrays
    va = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
    )

    # Chain of operations
    # 1. Add 10
    # 2. Multiply by 2
    # 3. Check if > 25
    # Each step should materialize the VirtualArray

    result1 = numpy_like.add(va, 10)  # [11, 12, 13, 14, 15]
    assert va.is_materialized

    result2 = result1 * 2  # [22, 24, 26, 28, 30]

    result3 = result2 > 25  # [False, False, True, True, True]
    np.testing.assert_array_equal(result3, np.array([False, False, True, True, True]))


def test_numpyarray_to_list(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.to_list(virtual_numpyarray) == ak.to_list(numpyarray)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_to_json(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.to_json(virtual_numpyarray) == ak.to_json(numpyarray)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_to_numpy(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert np.all(ak.to_numpy(virtual_numpyarray) == ak.to_numpy(numpyarray))
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_to_buffers(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    out1 = ak.to_buffers(numpyarray)
    out2 = ak.to_buffers(virtual_numpyarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert out1[2].keys() == out2[2].keys()
    for key in out1[2]:
        assert isinstance(out1[2][key], np.ndarray)
        assert isinstance(out2[2][key], VirtualArray)
        assert np.all(out1[2][key] == out2[2][key])


def test_numpyarray_is_valid(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.is_valid(virtual_numpyarray) == ak.is_valid(numpyarray)
    assert ak.validity_error(virtual_numpyarray) == ak.validity_error(numpyarray)


def test_numpyarray_zip(numpyarray, virtual_numpyarray):
    zip1 = ak.zip({"x": numpyarray, "y": numpyarray})
    zip2 = ak.zip({"x": virtual_numpyarray, "y": virtual_numpyarray})
    assert not zip2.layout.is_any_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_numpyarray_unzip(numpyarray, virtual_numpyarray):
    zip1 = ak.zip({"x": numpyarray, "y": numpyarray})
    zip2 = ak.zip({"x": virtual_numpyarray, "y": virtual_numpyarray})
    assert not zip2.layout.is_any_materialized
    unzip1 = ak.unzip(zip1)
    unzip2 = ak.unzip(zip2)
    assert not unzip2[0].layout.is_any_materialized
    assert not unzip2[1].layout.is_any_materialized
    assert ak.array_equal(ak.materialize(unzip2[0]), unzip1[0])
    assert ak.array_equal(ak.materialize(unzip2[1]), unzip1[1])


def test_numpyarray_concatenate(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.concatenate([numpyarray, numpyarray]),
        ak.concatenate([virtual_numpyarray, virtual_numpyarray]),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_where(numpyarray, virtual_numpyarray):
    numpyarray = ak.Array(numpyarray)
    virtual_numpyarray = ak.Array(virtual_numpyarray)
    assert not virtual_numpyarray.layout.is_any_materialized
    assert ak.array_equal(
        ak.where(numpyarray > 2, numpyarray, numpyarray + 100),
        ak.where(virtual_numpyarray > 2, virtual_numpyarray, virtual_numpyarray + 100),
    )
    assert virtual_numpyarray.layout.is_any_materialized
    assert virtual_numpyarray.layout.is_all_materialized


def test_numpyarray_unflatten(numpyarray, virtual_numpyarray):
    numpyarray = ak.Array(numpyarray)
    virtual_numpyarray = ak.Array(virtual_numpyarray)
    assert not virtual_numpyarray.layout.is_any_materialized
    assert ak.array_equal(
        ak.unflatten(numpyarray, [2, 3]),
        ak.unflatten(virtual_numpyarray, [2, 3]),
    )
    assert virtual_numpyarray.layout.is_any_materialized
    assert virtual_numpyarray.layout.is_all_materialized


def test_numpyarray_num(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.num(virtual_numpyarray, axis=0) == ak.num(numpyarray, axis=0)
    assert not virtual_numpyarray.is_any_materialized


def test_numpyarray_count(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.count(virtual_numpyarray, axis=0) == ak.count(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numppyarray_count_nonzero(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.count_nonzero(virtual_numpyarray, axis=0) == ak.count_nonzero(
        numpyarray, axis=0
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_sum(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.sum(virtual_numpyarray, axis=0) == ak.sum(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nansum(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nansum(virtual_numpyarray, axis=0) == ak.nansum(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_prod(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.prod(virtual_numpyarray, axis=0) == ak.prod(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanprod(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanprod(virtual_numpyarray, axis=0) == ak.nanprod(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_any(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.any(virtual_numpyarray, axis=0) == ak.any(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_all(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.all(virtual_numpyarray, axis=0) == ak.all(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_min(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.min(virtual_numpyarray, axis=0) == ak.min(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanmin(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanmin(virtual_numpyarray, axis=0) == ak.nanmin(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_max(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.max(virtual_numpyarray, axis=0) == ak.max(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanmax(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanmax(virtual_numpyarray, axis=0) == ak.nanmax(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argmin(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.argmin(virtual_numpyarray, axis=0) == ak.argmin(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanargmin(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanargmin(virtual_numpyarray, axis=0) == ak.nanargmin(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argmax(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.argmax(virtual_numpyarray, axis=0) == ak.argmax(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_nanargmax(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.nanargmax(virtual_numpyarray, axis=0) == ak.nanargmax(numpyarray, axis=0)
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_sort(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.sort(virtual_numpyarray, axis=0),
        ak.sort(numpyarray, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argsort(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.argsort(virtual_numpyarray, axis=0),
        ak.argsort(numpyarray, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_is_none(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.all(ak.is_none(virtual_numpyarray) == ak.is_none(numpyarray))
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_drop_none(numpy_like):
    array = ak.Array([1, None, 2, 3, None, 4, 5]).layout
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, -1, 1, 2, -1, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedOptionArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpy_array_pad_none(numpy_like):
    array = ak.Array([1, None, 2, 3, None, 4, 5]).layout
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, -1, 1, 2, -1, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedOptionArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(
        ak.pad_none(virtual_array, 10, axis=0), ak.pad_none(array, 10, axis=0)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_fill_none(numpy_like):
    array = ak.Array([1, None, 2, 3, None, 4, 5]).layout
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, -1, 1, 2, -1, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedOptionArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.fill_none(virtual_array, 100), ak.fill_none(array, 100))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_firsts(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.firsts(virtual_numpyarray, axis=0),
        ak.firsts(numpyarray, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_singletons(numpy_like):
    array = ak.Array([1, 2, 3, 4, 5]).layout
    virtual_index = ak.index.Index(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, 1, 2, 3, 4], dtype=np.int64),
        )
    )
    virtual_content = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    )
    virtual_array = ak.contents.IndexedArray(virtual_index, virtual_content)
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.singletons(virtual_array), ak.singletons(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_to_regular(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.to_regular(virtual_numpyarray, axis=0), ak.to_regular(numpyarray, axis=0)
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_broadcast_arrays(virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    out = ak.broadcast_arrays(5, virtual_numpyarray)
    assert ak.to_list(out[0]) == [5, 5, 5, 5, 5]
    assert not virtual_numpyarray.is_any_materialized
    assert not out[1].layout.is_any_materialized
    assert ak.to_list(out[1]) == ak.to_list(virtual_numpyarray)


def test_numpyarray_cartesian(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.cartesian([numpyarray, numpyarray], axis=0),
        ak.cartesian([virtual_numpyarray, virtual_numpyarray], axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argcartesian(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.argcartesian([numpyarray, numpyarray], axis=0),
        ak.argcartesian([virtual_numpyarray, virtual_numpyarray], axis=0),
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_combinations(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.combinations(numpyarray, 2, axis=0),
        ak.combinations(virtual_numpyarray, 2, axis=0),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_argcombinations(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.argcombinations(numpyarray, 2, axis=0),
        ak.argcombinations(virtual_numpyarray, 2, axis=0),
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_nan_to_none(numpy_like):
    array = ak.Array([1, np.nan, 2, 3, np.nan, 4, 5]).layout
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array(
                [1, np.nan, 2, 3, np.nan, 4, 5], dtype=np.float64
            ),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_none(virtual_array), ak.nan_to_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_nan_to_num(numpy_like):
    array = ak.Array([1, np.nan, 2, 3, np.nan, 4, 5]).layout
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(7,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array(
                [1, np.nan, 2, 3, np.nan, 4, 5], dtype=np.float64
            ),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_num(virtual_array), ak.nan_to_num(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_local_index(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.local_index(virtual_numpyarray, axis=0), ak.local_index(numpyarray, axis=0)
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_run_lengths(numpy_like):
    array = ak.Array([1, 1, 2, 3, 3, 3, 4, 5]).layout
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(8,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 1, 2, 3, 3, 3, 4, 5], dtype=np.int64),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.run_lengths(virtual_array), ak.run_lengths(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_round(numpy_like):
    array = ak.Array([1.234, 2.567, 3.499, 4.501]).layout
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(4,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.round(virtual_array), ak.round(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_isclose(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.isclose(virtual_numpyarray, numpyarray, rtol=1e-5, atol=1e-8),
        ak.isclose(numpyarray, numpyarray, rtol=1e-5, atol=1e-8),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_almost_equal(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.almost_equal(virtual_numpyarray, numpyarray),
        ak.almost_equal(numpyarray, numpyarray),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_real(numpy_like):
    array = ak.Array([1 + 2j, 3 + 4j, 5 + 6j]).layout
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(3,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.real(virtual_array), ak.real(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_imag(numpy_like):
    array = ak.Array([1 + 2j, 3 + 4j, 5 + 6j]).layout
    virtual_array = ak.contents.NumpyArray(
        VirtualArray(
            numpy_like,
            shape=(3,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
        )
    )
    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.imag(virtual_array), ak.imag(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_numpyarray_angle(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.angle(virtual_numpyarray, deg=True),
        ak.angle(numpyarray, deg=True),
    )
    assert virtual_numpyarray.is_any_materialized
    assert virtual_numpyarray.is_all_materialized


def test_numpyarray_zeros_like(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(ak.zeros_like(virtual_numpyarray), ak.zeros_like(numpyarray))
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_ones_like(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(ak.ones_like(virtual_numpyarray), ak.ones_like(numpyarray))
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_numpyarray_full_like(numpyarray, virtual_numpyarray):
    assert not virtual_numpyarray.is_any_materialized
    assert ak.array_equal(
        ak.full_like(virtual_numpyarray, 100), ak.full_like(numpyarray, 100)
    )
    assert not virtual_numpyarray.is_any_materialized
    assert not virtual_numpyarray.is_all_materialized


def test_listoffsetarray_to_list(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.to_list(virtual_listoffsetarray) == ak.to_list(listoffsetarray)
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_to_json(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.to_json(virtual_listoffsetarray) == ak.to_json(listoffsetarray)
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_to_numpy(listoffsetarray, virtual_listoffsetarray):
    # ListOffsetArray can't be directly converted to numpy for non-rectangular data
    # Test with flattened data instead
    assert not virtual_listoffsetarray.is_any_materialized
    flat_listoffsetarray = ak.flatten(listoffsetarray)
    flat_virtual_listoffsetarray = ak.flatten(virtual_listoffsetarray)
    assert np.all(
        ak.to_numpy(flat_virtual_listoffsetarray) == ak.to_numpy(flat_listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_to_buffers(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    out1 = ak.to_buffers(listoffsetarray)
    out2 = ak.to_buffers(virtual_listoffsetarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert set(out1[2].keys()) == set(out2[2].keys())
    for key in out1[2]:
        if isinstance(out2[2][key], VirtualArray):
            assert not out2[2][key].is_materialized
            assert np.all(out1[2][key] == out2[2][key])
            assert out2[2][key].is_materialized
        else:
            assert np.all(out1[2][key] == out2[2][key])


def test_listoffsetarray_is_valid(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.is_valid(virtual_listoffsetarray) == ak.is_valid(listoffsetarray)
    assert ak.validity_error(virtual_listoffsetarray) == ak.validity_error(
        listoffsetarray
    )


def test_listoffsetarray_zip(listoffsetarray, virtual_listoffsetarray):
    zip1 = ak.zip({"x": listoffsetarray, "y": listoffsetarray})
    zip2 = ak.zip({"x": virtual_listoffsetarray, "y": virtual_listoffsetarray})
    assert zip2.layout.is_any_materialized
    assert not zip2.layout.is_all_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_listoffsetarray_unzip(listoffsetarray, virtual_listoffsetarray):
    zip1 = ak.zip({"x": listoffsetarray, "y": listoffsetarray})
    zip2 = ak.zip({"x": virtual_listoffsetarray, "y": virtual_listoffsetarray})
    assert zip2.layout.is_any_materialized
    unzip1 = ak.unzip(zip1)
    unzip2 = ak.unzip(zip2)
    assert unzip2[0].layout.is_any_materialized
    assert unzip2[1].layout.is_any_materialized
    assert ak.array_equal(ak.materialize(unzip2[0]), unzip1[0])
    assert ak.array_equal(ak.materialize(unzip2[1]), unzip1[1])


def test_listoffsetarray_concatenate(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.concatenate([listoffsetarray, listoffsetarray]),
        ak.concatenate([virtual_listoffsetarray, virtual_listoffsetarray]),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_where(listoffsetarray, virtual_listoffsetarray):
    # We need to use ak.Array for the where function
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # For nested arrays, we need a compatible mask
    # Create a mask that's True for some elements in each list
    mask = ak.Array(
        [[True, False], [False, True], [True, False, True], [False, True, True]]
    )

    # Test with a conditional mask
    result1 = ak.where(mask, list_array, 999)
    result2 = ak.where(mask, virtual_list_array, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_listoffsetarray_flaten(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.flatten(virtual_listoffsetarray),
        ak.flatten(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_unflatten(listoffsetarray, virtual_listoffsetarray):
    # First flatten the arrays
    flat_list = ak.flatten(listoffsetarray)
    flat_virtual = ak.flatten(virtual_listoffsetarray)

    # Define counts for unflattening
    counts = np.array([2, 2, 3, 3])

    assert virtual_listoffsetarray.is_any_materialized

    # Unflatten and compare
    result1 = ak.unflatten(flat_list, counts)
    result2 = ak.unflatten(flat_virtual, counts)

    assert ak.array_equal(result1, result2)
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_num(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(ak.num(virtual_listoffsetarray) == ak.num(listoffsetarray))
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_count(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.count(virtual_listoffsetarray, axis=1) == ak.count(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_count_nonzero(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.count_nonzero(virtual_listoffsetarray, axis=1)
        == ak.count_nonzero(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_sum(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.sum(virtual_listoffsetarray, axis=1) == ak.sum(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nansum(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nansum(virtual_listoffsetarray, axis=1) == ak.nansum(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_prod(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.prod(virtual_listoffsetarray, axis=1) == ak.prod(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanprod(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nanprod(virtual_listoffsetarray, axis=1)
        == ak.nanprod(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_any(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.any(virtual_listoffsetarray, axis=1) == ak.any(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_all(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.all(virtual_listoffsetarray, axis=1) == ak.all(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_min(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.min(virtual_listoffsetarray, axis=1) == ak.min(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanmin(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nanmin(virtual_listoffsetarray, axis=1) == ak.nanmin(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_max(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.max(virtual_listoffsetarray, axis=1) == ak.max(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanmax(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(
        ak.nanmax(virtual_listoffsetarray, axis=1) == ak.nanmax(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argmin(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argmin(virtual_listoffsetarray), ak.argmin(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanargmin(numpy_like):
    # Create arrays with NaN values to test nanargmin
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmin(array, axis=1)
    result2 = ak.nanargmin(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_argmax(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argmax(virtual_listoffsetarray), ak.argmax(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nanargmax(numpy_like):
    # Create arrays with NaN values to test nanargmax
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmax(array, axis=1)
    result2 = ak.nanargmax(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_sort(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.sort(virtual_listoffsetarray),
        ak.sort(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argsort(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argsort(virtual_listoffsetarray),
        ak.argsort(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_is_none(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.all(ak.is_none(virtual_listoffsetarray) == ak.is_none(listoffsetarray))
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_drop_none(numpy_like):
    # Create a ListOffsetArray with some None values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListOffsetArray(ak.index.Index(offsets), indexed_content)

    # Create virtual versions
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), virtual_indexed_content
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_pad_none(numpy_like):
    # Create a regular ListOffsetArray
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Pad each list to length 5
    assert ak.array_equal(
        ak.pad_none(virtual_array, 5, axis=1), ak.pad_none(array, 5, axis=1)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_fill_none(numpy_like):
    # Create a ListOffsetArray with some None values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListOffsetArray(ak.index.Index(offsets), indexed_content)

    # Create virtual versions
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), virtual_indexed_content
    )

    assert not virtual_array.is_any_materialized
    # Fill None values with 999.0
    assert ak.array_equal(
        ak.fill_none(virtual_array, 999.0), ak.fill_none(array, 999.0)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_firsts(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.firsts(virtual_listoffsetarray),
        ak.firsts(listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_singletons(numpy_like):
    # Create a regular array to test
    offsets = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    content = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 1, 2, 3, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.singletons(virtual_array), ak.singletons(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_to_regular(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.to_regular(virtual_listoffsetarray, axis=0),
        ak.to_regular(listoffsetarray, axis=0),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_broadcast_arrays(virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    out = ak.broadcast_arrays(5, virtual_listoffsetarray)
    assert virtual_listoffsetarray.is_any_materialized
    assert out[1].layout.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized
    assert not out[1].layout.is_all_materialized
    assert ak.to_list(out[1]) == ak.to_list(virtual_listoffsetarray)


def test_listoffsetarray_cartesian(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.cartesian([listoffsetarray, listoffsetarray]),
        ak.cartesian([virtual_listoffsetarray, virtual_listoffsetarray]),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argcartesian(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.argcartesian([listoffsetarray, listoffsetarray]),
        ak.argcartesian([virtual_listoffsetarray, virtual_listoffsetarray]),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_combinations(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.combinations(virtual_listoffsetarray, 2, axis=1),
        ak.combinations(listoffsetarray, 2, axis=1),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_argcombinations(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.argcombinations(virtual_listoffsetarray, 2, axis=1),
        ak.argcombinations(listoffsetarray, 2, axis=1),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_nan_to_none(numpy_like):
    # Create a ListOffsetArray with NaN values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_none(virtual_array), ak.nan_to_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_nan_to_num(numpy_like):
    # Create a ListOffsetArray with NaN values
    offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_num(virtual_array), ak.nan_to_num(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_local_index(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    # For ListOffsetArray, we should use axis=1 to get indices within each list
    assert ak.array_equal(
        ak.local_index(virtual_listoffsetarray, axis=1),
        ak.local_index(listoffsetarray, axis=1),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_run_lengths(numpy_like):
    # Create a ListOffsetArray with repeated values for run_lengths test
    offsets = np.array([0, 3, 6], dtype=np.int64)
    content = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3, 6], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 1, 2, 3, 3, 3], dtype=np.int64),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    # Need to use axis=1 to check run lengths within each list
    assert ak.array_equal(ak.run_lengths(virtual_array), ak.run_lengths(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_round(numpy_like):
    # Create a ListOffsetArray with float values for rounding
    offsets = np.array([0, 2, 4], dtype=np.int64)
    content = np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.round(virtual_array), ak.round(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_isclose(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.isclose(virtual_listoffsetarray, listoffsetarray, rtol=1e-5, atol=1e-8),
        ak.isclose(listoffsetarray, listoffsetarray, rtol=1e-5, atol=1e-8),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_almost_equal(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.almost_equal(virtual_listoffsetarray, listoffsetarray),
        ak.almost_equal(listoffsetarray, listoffsetarray),
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_real(numpy_like):
    # Create a ListOffsetArray with complex values for real test
    offsets = np.array([0, 2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.real(virtual_array), ak.real(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_imag(numpy_like):
    # Create a ListOffsetArray with complex values for imag test
    offsets = np.array([0, 2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.imag(virtual_array), ak.imag(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_angle(numpy_like):
    # Create a ListOffsetArray with complex values for angle test
    offsets = np.array([0, 2, 4], dtype=np.int64)
    content = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    array = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array(
            [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
        ),
    )

    virtual_array = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(
        ak.angle(virtual_array, deg=True),
        ak.angle(array, deg=True),
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listoffsetarray_zeros_like(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.zeros_like(virtual_listoffsetarray), ak.zeros_like(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_ones_like(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.ones_like(virtual_listoffsetarray), ak.ones_like(listoffsetarray)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_full_like(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.full_like(virtual_listoffsetarray, 100), ak.full_like(listoffsetarray, 100)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert not virtual_listoffsetarray.is_all_materialized


# Additional tests for ListOffsetArray-specific operations


def test_listoffsetarray_flatten(listoffsetarray, virtual_listoffsetarray):
    assert not virtual_listoffsetarray.is_any_materialized
    assert ak.array_equal(
        ak.flatten(virtual_listoffsetarray, axis=1), ak.flatten(listoffsetarray, axis=1)
    )
    assert virtual_listoffsetarray.is_any_materialized
    assert virtual_listoffsetarray.is_all_materialized


def test_listoffsetarray_slicing(listoffsetarray, virtual_listoffsetarray):
    # Convert to ak.Array for slicing operations
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test slicing the outer dimension
    assert ak.array_equal(virtual_list_array[1:3], list_array[1:3])

    # Test indexing and then slicing inner dimension
    assert ak.array_equal(virtual_list_array[0][0:2], list_array[0][0:2])

    assert virtual_list_array.layout.is_any_materialized


def test_listoffsetarray_mask_operations(listoffsetarray, virtual_listoffsetarray):
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Create a boolean mask
    mask = ak.Array([True, False, True, False])

    # Test masking
    assert ak.array_equal(virtual_list_array[mask], list_array[mask])

    assert virtual_list_array.layout.is_any_materialized


def test_listoffsetarray_arithmetics(listoffsetarray, virtual_listoffsetarray):
    list_array = ak.Array(listoffsetarray)
    virtual_list_array = ak.Array(virtual_listoffsetarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test addition
    assert ak.array_equal(virtual_list_array + 10, list_array + 10)

    # Test multiplication
    assert ak.array_equal(virtual_list_array * 2, list_array * 2)

    # Test division
    assert ak.array_equal(virtual_list_array / 2, list_array / 2)

    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_listarray_to_list(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.to_list(virtual_listarray) == ak.to_list(listarray)
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_to_json(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.to_json(virtual_listarray) == ak.to_json(listarray)
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_to_numpy(listarray, virtual_listarray):
    # ListArray can't be directly converted to numpy for non-rectangular data
    # Test with flattened data instead
    assert not virtual_listarray.is_any_materialized
    flat_listarray = ak.flatten(listarray)
    flat_virtual_listarray = ak.flatten(virtual_listarray)
    assert np.all(ak.to_numpy(flat_virtual_listarray) == ak.to_numpy(flat_listarray))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_to_buffers(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    out1 = ak.to_buffers(listarray)
    out2 = ak.to_buffers(virtual_listarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert set(out1[2].keys()) == set(out2[2].keys())
    for key in out1[2]:
        if isinstance(out2[2][key], VirtualArray):
            assert not out2[2][key].is_materialized
            assert np.all(out1[2][key] == out2[2][key])
            assert out2[2][key].is_materialized
        else:
            assert np.all(out1[2][key] == out2[2][key])


def test_listarray_is_valid(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.is_valid(virtual_listarray) == ak.is_valid(listarray)
    assert ak.validity_error(virtual_listarray) == ak.validity_error(listarray)


def test_listarray_zip(listarray, virtual_listarray):
    zip1 = ak.zip({"x": listarray, "y": listarray})
    zip2 = ak.zip({"x": virtual_listarray, "y": virtual_listarray})
    assert zip2.layout.is_any_materialized
    assert not zip2.layout.is_all_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_listarray_unzip(listarray, virtual_listarray):
    zip1 = ak.zip({"x": listarray, "y": listarray})
    zip2 = ak.zip({"x": virtual_listarray, "y": virtual_listarray})
    assert zip2.layout.is_any_materialized
    unzip1 = ak.unzip(zip1)
    unzip2 = ak.unzip(zip2)
    assert unzip2[0].layout.is_any_materialized
    assert unzip2[1].layout.is_any_materialized
    assert ak.array_equal(ak.materialize(unzip2[0]), unzip1[0])
    assert ak.array_equal(ak.materialize(unzip2[1]), unzip1[1])


def test_listarray_concatenate(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.concatenate([listarray, listarray]),
        ak.concatenate([virtual_listarray, virtual_listarray]),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_where(listarray, virtual_listarray):
    # We need to use ak.Array for the where function
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # For nested arrays, we need a compatible mask
    # Create a mask that's True for some elements in each list
    mask = ak.Array(
        [[True, False], [False, True], [True, False, True], [False, True, True]]
    )

    # Test with a conditional mask
    result1 = ak.where(mask, list_array, 999)
    result2 = ak.where(mask, virtual_list_array, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_listarray_flatten(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.flatten(virtual_listarray, axis=1), ak.flatten(listarray, axis=1)
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_unflatten(listarray, virtual_listarray):
    # First flatten the arrays
    flat_list = ak.flatten(listarray)
    flat_virtual = ak.flatten(virtual_listarray)

    # Define counts for unflattening
    counts = np.array([2, 2, 3, 3])

    assert virtual_listarray.is_any_materialized

    # Unflatten and compare
    result1 = ak.unflatten(flat_list, counts)
    result2 = ak.unflatten(flat_virtual, counts)

    assert ak.array_equal(result1, result2)
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_num(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.num(virtual_listarray) == ak.num(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_count(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.count(virtual_listarray, axis=1) == ak.count(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_count_nonzero(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(
        ak.count_nonzero(virtual_listarray, axis=1)
        == ak.count_nonzero(listarray, axis=1)
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_sum(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.sum(virtual_listarray, axis=1) == ak.sum(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nansum(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.nansum(virtual_listarray, axis=1) == ak.nansum(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_prod(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.prod(virtual_listarray, axis=1) == ak.prod(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanprod(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(
        ak.nanprod(virtual_listarray, axis=1) == ak.nanprod(listarray, axis=1)
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_any(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.any(virtual_listarray, axis=1) == ak.any(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_all(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.all(virtual_listarray, axis=1) == ak.all(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_min(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.min(virtual_listarray, axis=1) == ak.min(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanmin(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.nanmin(virtual_listarray, axis=1) == ak.nanmin(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_max(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.max(virtual_listarray, axis=1) == ak.max(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanmax(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.nanmax(virtual_listarray, axis=1) == ak.nanmax(listarray, axis=1))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argmin(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.argmin(virtual_listarray), ak.argmin(listarray))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanargmin(numpy_like):
    # Create arrays with NaN values to test nanargmin
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmin(array, axis=1)
    result2 = ak.nanargmin(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_argmax(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.argmax(virtual_listarray), ak.argmax(listarray))
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_nanargmax(numpy_like):
    # Create arrays with NaN values to test nanargmax
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Axis=1 to test within each nested list
    result1 = ak.nanargmax(array, axis=1)
    result2 = ak.nanargmax(virtual_array, axis=1)
    assert ak.array_equal(result1, result2)
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_sort(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.sort(virtual_listarray),
        ak.sort(listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argsort(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.argsort(virtual_listarray),
        ak.argsort(listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_is_none(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.all(ak.is_none(virtual_listarray) == ak.is_none(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_drop_none(numpy_like):
    # Create a ListArray with some None values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), indexed_content
    )

    # Create virtual versions
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        virtual_indexed_content,
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_pad_none(numpy_like):
    # Create a regular ListArray
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Pad each list to length 5
    assert ak.array_equal(
        ak.pad_none(virtual_array, 5, axis=1), ak.pad_none(array, 5, axis=1)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_fill_none(numpy_like):
    # Create a ListArray with some None values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)

    # Create an IndexedOptionArray for the content that has None values
    index_data = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    content_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    index = ak.index.Index(index_data)
    content = ak.contents.NumpyArray(content_data)
    indexed_content = ak.contents.IndexedOptionArray(index, content)

    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), indexed_content
    )

    # Create virtual versions
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_indexed_content = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_index), ak.contents.NumpyArray(virtual_content)
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        virtual_indexed_content,
    )

    assert not virtual_array.is_any_materialized
    # Fill None values with 999.0
    assert ak.array_equal(
        ak.fill_none(virtual_array, 999.0), ak.fill_none(array, 999.0)
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_firsts(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.firsts(virtual_listarray),
        ak.firsts(listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_singletons(numpy_like):
    # Create a regular array to test
    starts = np.array([0, 1, 2, 3], dtype=np.int64)
    stops = np.array([1, 2, 3, 4], dtype=np.int64)
    content = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 1, 2, 3], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 2, 3, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.singletons(virtual_array), ak.singletons(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_to_regular(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.to_regular(virtual_listarray, axis=0),
        ak.to_regular(listarray, axis=0),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_broadcast_arrays(virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    out = ak.broadcast_arrays(5, virtual_listarray)
    assert virtual_listarray.is_any_materialized
    assert out[1].layout.is_any_materialized
    assert not virtual_listarray.is_all_materialized
    assert not out[1].layout.is_all_materialized
    assert ak.to_list(out[1]) == ak.to_list(virtual_listarray)


def test_listarray_cartesian(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.cartesian([listarray, listarray]),
        ak.cartesian([virtual_listarray, virtual_listarray]),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argcartesian(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.argcartesian([listarray, listarray]),
        ak.argcartesian([virtual_listarray, virtual_listarray]),
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_combinations(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.combinations(virtual_listarray, 2, axis=1),
        ak.combinations(listarray, 2, axis=1),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_argcombinations(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    # Need to use axis=1 for nested structures
    assert ak.array_equal(
        ak.argcombinations(virtual_listarray, 2, axis=1),
        ak.argcombinations(listarray, 2, axis=1),
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_nan_to_none(numpy_like):
    # Create a ListArray with NaN values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_none(virtual_array), ak.nan_to_none(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_nan_to_num(numpy_like):
    # Create a ListArray with NaN values
    starts = np.array([0, 2, 4, 7], dtype=np.int64)
    stops = np.array([2, 4, 7, 10], dtype=np.int64)
    content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4, 7, 10], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.nan_to_num(virtual_array), ak.nan_to_num(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_local_index(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    # For ListArray, we should use axis=1 to get indices within each list
    assert ak.array_equal(
        ak.local_index(virtual_listarray, axis=1),
        ak.local_index(listarray, axis=1),
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_run_lengths(numpy_like):
    # Create a ListArray with repeated values for run_lengths test
    starts = np.array([0, 3], dtype=np.int64)
    stops = np.array([3, 6], dtype=np.int64)
    content = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([3, 6], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 1, 2, 3, 3, 3], dtype=np.int64),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    # Need to use axis=1 to check run lengths within each list
    assert ak.array_equal(ak.run_lengths(virtual_array), ak.run_lengths(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_round(numpy_like):
    # Create a ListArray with float values for rounding
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 4], dtype=np.int64)
    content = np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.round(virtual_array), ak.round(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_isclose(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.isclose(virtual_listarray, listarray, rtol=1e-5, atol=1e-8),
        ak.isclose(listarray, listarray, rtol=1e-5, atol=1e-8),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_almost_equal(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.almost_equal(virtual_listarray, listarray),
        ak.almost_equal(listarray, listarray),
    )
    assert virtual_listarray.is_any_materialized
    assert virtual_listarray.is_all_materialized


def test_listarray_real(numpy_like):
    # Create a ListArray with complex values for real test
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.real(virtual_array), ak.real(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_imag(numpy_like):
    # Create a ListArray with complex values for imag test
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 3], dtype=np.int64)
    content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 3], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(ak.imag(virtual_array), ak.imag(array))
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_angle(numpy_like):
    # Create a ListArray with complex values for angle test
    starts = np.array([0, 2], dtype=np.int64)
    stops = np.array([2, 4], dtype=np.int64)
    content = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    array = ak.contents.ListArray(
        ak.index.Index(starts), ak.index.Index(stops), ak.contents.NumpyArray(content)
    )

    # Create virtual version
    virtual_starts = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2], dtype=np.int64),
    )

    virtual_stops = VirtualArray(
        numpy_like,
        shape=(2,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([2, 4], dtype=np.int64),
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array(
            [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
        ),
    )

    virtual_array = ak.contents.ListArray(
        ak.index.Index(virtual_starts),
        ak.index.Index(virtual_stops),
        ak.contents.NumpyArray(virtual_content),
    )

    assert not virtual_array.is_any_materialized
    assert ak.array_equal(
        ak.angle(virtual_array, deg=True),
        ak.angle(array, deg=True),
    )
    assert virtual_array.is_any_materialized
    assert virtual_array.is_all_materialized


def test_listarray_zeros_like(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.zeros_like(virtual_listarray), ak.zeros_like(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_ones_like(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(ak.ones_like(virtual_listarray), ak.ones_like(listarray))
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_full_like(listarray, virtual_listarray):
    assert not virtual_listarray.is_any_materialized
    assert ak.array_equal(
        ak.full_like(virtual_listarray, 100), ak.full_like(listarray, 100)
    )
    assert virtual_listarray.is_any_materialized
    assert not virtual_listarray.is_all_materialized


def test_listarray_slicing(listarray, virtual_listarray):
    # Convert to ak.Array for slicing operations
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test slicing the outer dimension
    assert ak.array_equal(virtual_list_array[1:3], list_array[1:3])

    # Test indexing and then slicing inner dimension
    assert ak.array_equal(virtual_list_array[0][0:2], list_array[0][0:2])

    assert virtual_list_array.layout.is_any_materialized


def test_listarray_mask_operations(listarray, virtual_listarray):
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Create a boolean mask
    mask = ak.Array([True, False, True, False])

    # Test masking
    assert ak.array_equal(virtual_list_array[mask], list_array[mask])

    assert virtual_list_array.layout.is_any_materialized


def test_listarray_arithmetics(listarray, virtual_listarray):
    list_array = ak.Array(listarray)
    virtual_list_array = ak.Array(virtual_listarray)

    assert not virtual_list_array.layout.is_any_materialized

    # Test addition
    assert ak.array_equal(virtual_list_array + 10, list_array + 10)

    # Test multiplication
    assert ak.array_equal(virtual_list_array * 2, list_array * 2)

    # Test division
    assert ak.array_equal(virtual_list_array / 2, list_array / 2)

    assert virtual_list_array.layout.is_any_materialized
    assert virtual_list_array.layout.is_all_materialized


def test_recordarray_to_list(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.to_list(virtual_recordarray) == ak.to_list(recordarray)
    assert virtual_recordarray.is_any_materialized
    assert virtual_recordarray.is_all_materialized


def test_recordarray_to_json(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.to_json(virtual_recordarray) == ak.to_json(recordarray)
    assert virtual_recordarray.is_any_materialized
    assert virtual_recordarray.is_all_materialized


def test_recordarray_to_buffers(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    out1 = ak.to_buffers(recordarray)
    out2 = ak.to_buffers(virtual_recordarray)
    # form
    assert out1[0] == out2[0]
    # length
    assert out1[1] == out2[1]
    # container
    assert set(out1[2].keys()) == set(out2[2].keys())
    for key in out1[2]:
        if isinstance(out2[2][key], VirtualArray):
            assert not out2[2][key].is_materialized
            assert np.all(out1[2][key] == out2[2][key])
            assert out2[2][key].is_materialized
        else:
            assert np.all(out1[2][key] == out2[2][key])


def test_recordarray_is_valid(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.is_valid(virtual_recordarray) == ak.is_valid(recordarray)
    assert ak.validity_error(virtual_recordarray) == ak.validity_error(recordarray)


def test_recordarray_field_access(recordarray, virtual_recordarray):
    # Test accessing fields directly
    assert not virtual_recordarray.is_any_materialized

    # Access x field (ListOffsetArray)
    x_field = virtual_recordarray["x"]
    assert isinstance(x_field, ak.contents.ListOffsetArray)
    assert not x_field.is_any_materialized

    # Access y field (NumpyArray)
    y_field = virtual_recordarray["y"]
    assert isinstance(y_field, ak.contents.NumpyArray)
    assert not y_field.data.is_materialized

    # Verify field values
    assert ak.to_list(x_field) == ak.to_list(recordarray["x"])
    assert ak.to_list(y_field) == ak.to_list(recordarray["y"])

    # Check materialization state after access
    assert x_field.is_any_materialized
    assert y_field.data.is_materialized


def test_recordarray_zip(recordarray, virtual_recordarray):
    # Test zipping the RecordArray
    zip1 = ak.zip({"record": recordarray})
    zip2 = ak.zip({"record": virtual_recordarray})

    assert not virtual_recordarray.is_any_materialized
    assert zip1.fields == zip2.fields
    assert ak.array_equal(ak.materialize(zip2), zip1)


def test_recordarray_concatenate(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    result = ak.concatenate([virtual_recordarray, virtual_recordarray])
    expected = ak.concatenate([recordarray, recordarray])
    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_where_x_field(recordarray, virtual_recordarray):
    # Test where operation on the x field (ListOffsetArray)
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Create a mask compatible with nested list structure in x field
    mask = ak.Array(
        [[True, False], [False, True], [True, False, True], [False, True, True], []]
    )

    # Apply where operation on the x field
    result1 = ak.where(mask, record_array.x, 999)
    result2 = ak.where(mask, virtual_record_array.x, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_where_y_field(recordarray, virtual_recordarray):
    # Test where operation on the y field (NumpyArray)
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Create a simple mask for the y field
    mask = ak.Array([True, False, True, False, True])

    # Apply where operation on the y field
    result1 = ak.where(mask, record_array.y, 999)
    result2 = ak.where(mask, virtual_record_array.y, 999)

    assert ak.array_equal(result1, result2)
    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_count_x_field(recordarray, virtual_recordarray):
    # Test count on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.count(recordarray["x"], axis=1)

    result = ak.count(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_count_y_field(recordarray, virtual_recordarray):
    # Test count on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.count(recordarray["y"])

    result = ak.count(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_count_nonzero_x_field(recordarray, virtual_recordarray):
    # Test count_nonzero on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.count_nonzero(recordarray["x"], axis=1)

    result = ak.count_nonzero(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_count_nonzero_y_field(recordarray, virtual_recordarray):
    # Test count_nonzero on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.count_nonzero(recordarray["y"])

    result = ak.count_nonzero(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sum_x_field(recordarray, virtual_recordarray):
    # Test sum on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.sum(recordarray["x"], axis=1)

    result = ak.sum(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sum_y_field(recordarray, virtual_recordarray):
    # Test sum on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.sum(recordarray["y"])

    result = ak.sum(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_any_x_field(recordarray, virtual_recordarray):
    # Test any on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.any(recordarray["x"], axis=1)

    result = ak.any(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_any_y_field(recordarray, virtual_recordarray):
    # Test any on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.any(recordarray["y"])

    result = ak.any(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_all_x_field(recordarray, virtual_recordarray):
    # Test all on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.all(recordarray["x"], axis=1)

    result = ak.all(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_all_y_field(recordarray, virtual_recordarray):
    # Test all on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.all(recordarray["y"])

    result = ak.all(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_min_x_field(recordarray, virtual_recordarray):
    # Test min on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.min(recordarray["x"], axis=1)

    result = ak.min(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_min_y_field(recordarray, virtual_recordarray):
    # Test min on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.min(recordarray["y"])

    result = ak.min(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_max_x_field(recordarray, virtual_recordarray):
    # Test max on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.max(recordarray["x"], axis=1)

    result = ak.max(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_max_y_field(recordarray, virtual_recordarray):
    # Test max on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.max(recordarray["y"])

    result = ak.max(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmin_x_field(recordarray, virtual_recordarray):
    # Test argmin on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.argmin(recordarray["x"], axis=1)

    result = ak.argmin(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmin_y_field(recordarray, virtual_recordarray):
    # Test argmin on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.argmin(recordarray["y"])

    result = ak.argmin(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmax_x_field(recordarray, virtual_recordarray):
    # Test argmax on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.argmax(recordarray["x"], axis=1)

    result = ak.argmax(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argmax_y_field(recordarray, virtual_recordarray):
    # Test argmax on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.argmax(recordarray["y"])

    result = ak.argmax(y_field)
    assert result == expected

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sort_x_field(recordarray, virtual_recordarray):
    # Test sort on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.sort(recordarray["x"], axis=1)

    result = ak.sort(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_sort_y_field(recordarray, virtual_recordarray):
    # Test sort on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.sort(recordarray["y"])

    result = ak.sort(y_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argsort_x_field(recordarray, virtual_recordarray):
    # Test argsort on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.argsort(recordarray["x"], axis=1)

    result = ak.argsort(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_argsort_y_field(recordarray, virtual_recordarray):
    # Test argsort on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.argsort(recordarray["y"])

    result = ak.argsort(y_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_is_none(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized
    assert ak.array_equal(ak.is_none(virtual_recordarray), ak.is_none(recordarray))
    assert virtual_recordarray.is_any_materialized


def test_recordarray_local_index_x_field(recordarray, virtual_recordarray):
    # Test local_index on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.local_index(recordarray["x"], axis=1)

    result = ak.local_index(x_field, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_local_index_y_field(recordarray, virtual_recordarray):
    # Test local_index on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.local_index(recordarray["y"])

    result = ak.local_index(y_field)
    assert ak.array_equal(result, expected)
    assert not virtual_recordarray.is_any_materialized


def test_recordarray_combinations_x_field(recordarray, virtual_recordarray):
    # Test combinations on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.combinations(recordarray["x"], 2, axis=1)

    result = ak.combinations(x_field, 2, axis=1)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_combinations_y_field(recordarray, virtual_recordarray):
    # Test combinations on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.combinations(recordarray["y"], 2, axis=0)

    result = ak.combinations(y_field, 2, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_nan_to_none_x_field(numpy_like):
    # Create a RecordArray with NaN values in the x field
    offsets = np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)
    x_content = np.array(
        [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan], dtype=np.float64
    )
    y_content = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10, 10], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8, 9.9, np.nan],
            dtype=np.float64,
        ),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test nan_to_none on x field
    assert not virtual_array.is_any_materialized

    result_x = ak.nan_to_none(virtual_array["x"])
    expected_x = ak.nan_to_none(array["x"])

    assert ak.array_equal(result_x, expected_x)
    assert virtual_array.is_any_materialized


def test_recordarray_nan_to_none_y_field(numpy_like):
    # Create a RecordArray with NaN values in the y field
    offsets = np.array([0, 2, 4, 7, 10, 10], dtype=np.int64)
    x_content = np.array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
    )
    y_content = np.array([0.1, np.nan, 0.3, np.nan, 0.5], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10, 10], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array(
            [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], dtype=np.float64
        ),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, np.nan, 0.3, np.nan, 0.5], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test nan_to_none on y field
    assert not virtual_array.is_any_materialized

    result_y = ak.nan_to_none(virtual_array["y"])
    expected_y = ak.nan_to_none(array["y"])

    assert ak.array_equal(result_y, expected_y)
    assert virtual_array.is_any_materialized


def test_recordarray_zeros_like(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.zeros_like(virtual_recordarray)
    expected = ak.zeros_like(recordarray)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_ones_like(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.ones_like(virtual_recordarray)
    expected = ak.ones_like(recordarray)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_full_like(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.full_like(virtual_recordarray, 100)
    expected = ak.full_like(recordarray, 100)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_slicing(recordarray, virtual_recordarray):
    # Convert to ak.Array for slicing operations
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test slicing the record array
    assert ak.array_equal(virtual_record_array[1:3], record_array[1:3])

    # Test slicing a field
    assert ak.array_equal(virtual_record_array.x[1:3], record_array.x[1:3])
    assert ak.array_equal(virtual_record_array.y[1:3], record_array.y[1:3])

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_mask_operations(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Create a boolean mask
    mask = ak.Array([True, False, True, False, True])

    # Test masking the record array
    assert ak.array_equal(virtual_record_array[mask], record_array[mask])

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_arithmetics_x_field(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test addition on the x field
    assert ak.array_equal(virtual_record_array.x + 10, record_array.x + 10)

    # Test multiplication on the x field
    assert ak.array_equal(virtual_record_array.x * 2, record_array.x * 2)

    # Test division on the x field
    assert ak.array_equal(virtual_record_array.x / 2, record_array.x / 2)

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_arithmetics_y_field(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test addition on the y field
    assert ak.array_equal(virtual_record_array.y + 10, record_array.y + 10)

    # Test multiplication on the y field
    assert ak.array_equal(virtual_record_array.y * 2, record_array.y * 2)

    # Test division on the y field
    assert ak.array_equal(virtual_record_array.y / 2, record_array.y / 2)

    assert virtual_record_array.layout.is_any_materialized


def test_recordarray_firsts_x_field(recordarray, virtual_recordarray):
    # Test firsts on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.firsts(recordarray["x"])

    result = ak.firsts(x_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_firsts_y_field(recordarray, virtual_recordarray):
    # Test firsts on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.firsts(recordarray["y"], axis=0)

    result = ak.firsts(y_field, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_flatten_x_field(recordarray, virtual_recordarray):
    # Test flatten on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.flatten(recordarray["x"])

    result = ak.flatten(x_field)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_to_regular_x_field(recordarray, virtual_recordarray):
    # Test to_regular on the x field (ListOffsetArray)
    assert not virtual_recordarray.is_any_materialized

    x_field = virtual_recordarray["x"]
    expected = ak.to_regular(recordarray["x"], axis=0)

    result = ak.to_regular(x_field, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_to_regular_y_field(recordarray, virtual_recordarray):
    # Test to_regular on the y field (NumpyArray)
    assert not virtual_recordarray.is_any_materialized

    y_field = virtual_recordarray["y"]
    expected = ak.to_regular(recordarray["y"], axis=0)

    result = ak.to_regular(y_field, axis=0)
    assert ak.array_equal(result, expected)

    assert virtual_recordarray.is_any_materialized


def test_recordarray_run_lengths_x_field(numpy_like):
    # Create a RecordArray with repeated values in the x field for run_lengths test
    offsets = np.array([0, 3, 6, 6], dtype=np.int64)
    x_content = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3, 6, 6], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([1, 1, 2, 3, 3, 3], dtype=np.int64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test run_lengths on x field
    assert not virtual_array.is_any_materialized

    result = ak.run_lengths(virtual_array["x"])
    expected = ak.run_lengths(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_run_lengths_y_field(numpy_like):
    # Create a RecordArray with repeated values in the y field for run_lengths test
    offsets = np.array([0, 3, 6, 6], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)
    y_content = np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 3, 6, 6], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test run_lengths on y field
    assert not virtual_array.is_any_materialized

    result = ak.run_lengths(virtual_array["y"])
    expected = ak.run_lengths(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_round_x_field(numpy_like):
    # Create a RecordArray with float values in the x field for rounding
    offsets = np.array([0, 2, 4, 4], dtype=np.int64)
    x_content = np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 4], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test round on x field
    assert not virtual_array.is_any_materialized

    result = ak.round(virtual_array["x"])
    expected = ak.round(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_round_y_field(numpy_like):
    # Create a RecordArray with float values in the y field for rounding
    offsets = np.array([0, 2, 4, 4], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    y_content = np.array([1.234, 2.567, 3.499], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 4], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.234, 2.567, 3.499], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test round on y field
    assert not virtual_array.is_any_materialized

    result = ak.round(virtual_array["y"])
    expected = ak.round(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_isclose(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.isclose(virtual_recordarray, recordarray, rtol=1e-5, atol=1e-8)
    expected = ak.isclose(recordarray, recordarray, rtol=1e-5, atol=1e-8)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_almost_equal(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    result = ak.almost_equal(virtual_recordarray, recordarray)
    expected = ak.almost_equal(recordarray, recordarray)

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_real_x_field(numpy_like):
    # Create a RecordArray with complex values in the x field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test real on x field
    assert not virtual_array.is_any_materialized

    result = ak.real(virtual_array["x"])
    expected = ak.real(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_real_y_field(numpy_like):
    # Create a RecordArray with complex values in the y field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    y_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3], dtype=np.float64),
    )
    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test real on y field
    assert not virtual_array.is_any_materialized

    result = ak.real(virtual_array["y"])
    expected = ak.real(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_imag_x_field(numpy_like):
    # Create a RecordArray with complex values in the x field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test imag on x field
    assert not virtual_array.is_any_materialized

    result = ak.imag(virtual_array["x"])
    expected = ak.imag(array["x"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_imag_y_field(numpy_like):
    # Create a RecordArray with complex values in the y field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    y_content = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test imag on y field
    assert not virtual_array.is_any_materialized

    result = ak.imag(virtual_array["y"])
    expected = ak.imag(array["y"])

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_angle_x_field(numpy_like):
    # Create a RecordArray with complex values in the x field
    offsets = np.array([0, 2, 4, 4], dtype=np.int64)
    x_content = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 4], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array(
            [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
        ),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test angle on x field
    assert not virtual_array.is_any_materialized

    result = ak.angle(virtual_array["x"], deg=True)
    expected = ak.angle(array["x"], deg=True)

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_angle_y_field(numpy_like):
    # Create a RecordArray with complex values in the y field
    offsets = np.array([0, 2, 3, 3], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    y_content = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(x_content)
    )
    y_field = ak.contents.NumpyArray(y_content)
    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual version
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(4,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 3, 3], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3], dtype=np.float64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.complex128),
        generator=lambda: np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_x_content)
    )

    virtual_y_field = ak.contents.NumpyArray(virtual_y_content)

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test angle on y field
    assert not virtual_array.is_any_materialized

    result = ak.angle(virtual_array["y"], deg=True)
    expected = ak.angle(array["y"], deg=True)

    assert ak.array_equal(result, expected)
    assert virtual_array.is_any_materialized


def test_recordarray_fields(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    # Test fields property
    assert virtual_recordarray.fields == recordarray.fields
    assert not virtual_recordarray.is_any_materialized


def test_recordarray_with_field(recordarray, virtual_recordarray, numpy_like):
    assert not virtual_recordarray.is_any_materialized

    # Create a new field to add
    new_field = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
    )

    # Test with_field method which adds a new field
    result = ak.with_field(virtual_recordarray, ak.contents.NumpyArray(new_field), "z")
    expected = ak.with_field(
        recordarray, ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])), "z"
    )

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_without_field(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    # Test without_field method which removes a field
    result = ak.without_field(virtual_recordarray, "y")
    expected = ak.without_field(recordarray, "y")

    assert ak.array_equal(result, expected)
    assert virtual_recordarray.is_any_materialized


def test_recordarray_broadcast_arrays(recordarray, virtual_recordarray):
    assert not virtual_recordarray.is_any_materialized

    # Test broadcast_arrays with a RecordArray
    out = ak.broadcast_arrays(5, virtual_recordarray)
    assert len(out) == 2
    # The virtual_recordarray should stay virtual until accessed
    assert not virtual_recordarray.is_any_materialized
    assert not out[1].layout.is_any_materialized

    # Verify content after materialization
    assert ak.array_equal(out[1], recordarray)


def test_recordarray_with_custom_generator(numpy_like):
    # Create a RecordArray with a more complex generator function

    # Define generator functions that compute values
    def compute_offsets():
        # Compute offsets dynamically
        return np.array([0, 2, 5, 9, 10], dtype=np.int64)

    def compute_content():
        # Compute content with a formula
        return np.array([i**2 for i in range(10)], dtype=np.float64)

    def compute_y_values():
        # Compute y values with a formula
        return np.array([np.sin(i) for i in range(5)], dtype=np.float64)

    # Create virtual arrays with these generators
    virtual_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=compute_offsets,
    )

    virtual_content = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.float64),
        generator=compute_content,
    )

    virtual_y = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.float64),
        generator=compute_y_values,
    )

    # Create the RecordArray
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_offsets), ak.contents.NumpyArray(virtual_content)
    )

    y_field = ak.contents.NumpyArray(virtual_y)

    virtual_array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create the expected array for comparison
    offsets = np.array([0, 2, 5, 9, 10], dtype=np.int64)
    content = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], dtype=np.float64)
    y_values = np.array(
        [
            0.0,
            0.8414709848078965,
            0.9092974268256817,
            0.1411200080598672,
            -0.7568024953079282,
        ],
        dtype=np.float64,
    )

    x_field_regular = ak.contents.ListOffsetArray(
        ak.index.Index(offsets), ak.contents.NumpyArray(content)
    )

    y_field_regular = ak.contents.NumpyArray(y_values)

    regular_array = ak.contents.RecordArray(
        [x_field_regular, y_field_regular], ["x", "y"]
    )

    # Test operations
    assert not virtual_array.is_any_materialized

    # Check to_list
    assert ak.to_list(virtual_array) == ak.to_list(regular_array)

    # Check field access and operations
    assert ak.array_equal(
        ak.sum(virtual_array["x"], axis=1), ak.sum(regular_array["x"], axis=1)
    )
    assert ak.array_equal(ak.min(virtual_array["y"]), ak.min(regular_array["y"]))

    assert virtual_array.is_any_materialized


def test_recordarray_with_none_values(numpy_like):
    # Create a RecordArray with None values in both fields

    # Create x field with None values
    x_offsets = np.array([0, 2, 4, 7, 10], dtype=np.int64)
    x_index = np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64)
    x_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64)

    # Create y field with None values
    y_index = np.array([0, -1, 1, -1, 2], dtype=np.int64)
    y_content = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    # Create regular array
    x_field = ak.contents.ListOffsetArray(
        ak.index.Index(x_offsets),
        ak.contents.IndexedOptionArray(
            ak.index.Index(x_index), ak.contents.NumpyArray(x_content)
        ),
    )

    y_field = ak.contents.IndexedOptionArray(
        ak.index.Index(y_index), ak.contents.NumpyArray(y_content)
    )

    array = ak.contents.RecordArray([x_field, y_field], ["x", "y"])

    # Create virtual versions
    virtual_x_offsets = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, 2, 4, 7, 10], dtype=np.int64),
    )

    virtual_x_index = VirtualArray(
        numpy_like,
        shape=(10,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2, 3, -1, 4, 5, -1], dtype=np.int64),
    )

    virtual_x_content = VirtualArray(
        numpy_like,
        shape=(6,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], dtype=np.float64),
    )

    virtual_y_index = VirtualArray(
        numpy_like,
        shape=(5,),
        dtype=np.dtype(np.int64),
        generator=lambda: np.array([0, -1, 1, -1, 2], dtype=np.int64),
    )

    virtual_y_content = VirtualArray(
        numpy_like,
        shape=(3,),
        dtype=np.dtype(np.float64),
        generator=lambda: np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    virtual_x_field = ak.contents.ListOffsetArray(
        ak.index.Index(virtual_x_offsets),
        ak.contents.IndexedOptionArray(
            ak.index.Index(virtual_x_index), ak.contents.NumpyArray(virtual_x_content)
        ),
    )

    virtual_y_field = ak.contents.IndexedOptionArray(
        ak.index.Index(virtual_y_index), ak.contents.NumpyArray(virtual_y_content)
    )

    virtual_array = ak.contents.RecordArray(
        [virtual_x_field, virtual_y_field], ["x", "y"]
    )

    # Test operations with None values
    assert not virtual_array.is_any_materialized

    # Check to_list
    assert ak.to_list(virtual_array) == ak.to_list(array)

    # Test drop_none
    assert ak.array_equal(ak.drop_none(virtual_array), ak.drop_none(array))

    # Test fill_none
    assert ak.array_equal(ak.fill_none(virtual_array, 999), ak.fill_none(array, 999))

    # Test is_none
    assert ak.array_equal(ak.is_none(virtual_array), ak.is_none(array))

    assert virtual_array.is_any_materialized


def test_recordarray_advanced_indexing(recordarray, virtual_recordarray):
    record_array = ak.Array(recordarray)
    virtual_record_array = ak.Array(virtual_recordarray)

    assert not virtual_record_array.layout.is_any_materialized

    # Test slicing with step
    slice_result = virtual_record_array[::2]
    expected_slice = record_array[::2]
    assert ak.array_equal(slice_result, expected_slice)

    # Test fancy indexing with array of indices
    indices = np.array([3, 1, 2])
    fancy_result = virtual_record_array[indices]
    expected_fancy = record_array[indices]
    assert ak.array_equal(fancy_result, expected_fancy)

    # Test boolean masking
    mask = np.array([True, False, True, False, True])
    mask_result = virtual_record_array[mask]
    expected_mask = record_array[mask]
    assert ak.array_equal(mask_result, expected_mask)

    assert virtual_record_array.layout.is_any_materialized
