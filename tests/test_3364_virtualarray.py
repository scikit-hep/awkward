from __future__ import annotations

import numpy as np
import pytest

from awkward._nplikes.numpy import Numpy
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.virtual import VirtualArray


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


# Test initialization
def test_init_valid(numpy_like, simple_array_generator):
    va = VirtualArray(
        numpy_like, shape=(5,), dtype=np.int64, generator=simple_array_generator
    )
    assert va.shape == (5,)
    assert va.dtype == np.dtype(np.int64)
    assert not va.is_materialized


def test_init_invalid_nplike():
    with pytest.raises(TypeError, match=r"Only numpy and cupy nplikes are supported"):
        VirtualArray(
            "not_an_nplike",
            shape=(5,),
            dtype=np.int64,
            generator=lambda: np.array([1, 2, 3, 4, 5]),
        )


def test_init_invalid_shape():
    nplike = Numpy.instance()
    with pytest.raises(TypeError, match=r"supports only shapes of integer dimensions"):
        VirtualArray(
            nplike,
            shape=("not_an_integer", 5),
            dtype=np.int64,
            generator=lambda: np.array([[1, 2, 3, 4, 5]]),
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
    assert virtual_array.nbytes == 0


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
        TypeError,
        match=r"had shape \(5,\) before materialization while the materialized array has shape \(3,\)",
    ):
        va = VirtualArray(
            numpy_like,
            shape=(5,),
            dtype=np.int64,
            generator=lambda: np.array([1, 2, 3]),
        )
        va.materialize()


def test_materialize_dtype_mismatch(numpy_like):
    # Generator returns array with different dtype than declared
    with pytest.raises(
        TypeError,
        match=r"had dtype int64 before materialization while the materialized array has dtype float64",
    ):
        va = VirtualArray(
            numpy_like,
            shape=(3,),
            dtype=np.int64,
            generator=lambda: np.array([1.0, 2.0, 3.0]),
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
    assert virtual_array.generator is simple_array_generator


# Test nplike property
def test_nplike(virtual_array, numpy_like):
    assert virtual_array.nplike is numpy_like


def test_nplike_invalid():
    va = VirtualArray(
        Numpy.instance(),
        shape=(5,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3, 4, 5]),
    )
    va._nplike = "not_an_nplike"  # Directly modify to create an invalid state
    with pytest.raises(TypeError, match=r"Only numpy and cupy nplikes are supported"):
        _ = va.nplike


# Test copy
def test_copy(virtual_array):
    copy = virtual_array.copy()
    assert isinstance(copy, VirtualArray)
    assert copy.shape == virtual_array.shape
    assert copy.dtype == virtual_array.dtype
    assert copy.is_materialized  # Copy should be materialized
    assert id(copy) != id(virtual_array)  # Different objects


# Test tolist
def test_tolist(virtual_array):
    assert virtual_array.tolist() == [1, 2, 3, 4, 5]


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
        nplike, shape=(5,), dtype=np.int64, generator=lambda: np.array([1, 2, 3, 4, 5])
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
        nplike, shape=(), dtype=np.int64, generator=lambda: np.array(0)
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
        nplike, shape=(), dtype=np.int64, generator=lambda: np.array(42)
    )
    with pytest.raises(TypeError, match=r"len\(\) of unsized object"):
        len(scalar_va)


# Test __iter__
def test_iter(virtual_array):
    assert list(virtual_array) == [1, 2, 3, 4, 5]


# Test __dlpack__ and __dlpack_device__
def test_dlpack_device(virtual_array):
    with pytest.raises(RuntimeError, match=r"cannot realise an unknown value"):
        virtual_array.__dlpack_device__()


def test_dlpack(virtual_array):
    with pytest.raises(RuntimeError, match=r"cannot realise an unknown value"):
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
        nplike, shape=(3,), dtype=np.int64, generator=lambda: np.array([1, 2, 3])
    )
    va2 = VirtualArray(
        nplike, shape=(2,), dtype=np.int64, generator=lambda: np.array([4, 5])
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
        generator=lambda: np.array([1.1, 2.5, 3.7, 4.2, 5.9]),
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
        generator=lambda: np.array([1.1, np.nan, 3.3, np.nan, 5.5]),
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
        generator=lambda: np.array([[1, 2, 3], [4, 5, 6]]),
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

    # Before materialization, nbytes should be 0
    assert va.nbytes == 0

    # Access just one element to check if full materialization happens
    element = va[0, 0]
    assert element == 1.0

    # Now the array should be materialized
    assert va.nbytes > 0
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
        generator=lambda: np.array([10, 20, 30]),
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
        generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j]),
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
        generator=lambda: np.array([1, 2, 3, 4, 5]),
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
