from __future__ import annotations

import numpy as np
import pytest

from awkward._backends.dispatch import backend_of_obj
from awkward._nplikes.dispatch import nplike_of_obj
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.virtual import VirtualArray, materialize_if_virtual


@pytest.fixture
def numpy_like():
    return Numpy.instance()


@pytest.fixture(
    params=[
        None,  # Don't use shape_generator
        "use_shape_generator",  # Use shape_generator
    ]
)
def shape_generator_param(request):
    return request.param


@pytest.fixture
def simple_array_generator():
    return lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64)


@pytest.fixture
def virtual_array(numpy_like, simple_array_generator, shape_generator_param):
    if shape_generator_param is None:
        return VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=simple_array_generator,
        )
    else:
        return VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=simple_array_generator,
            shape_generator=lambda: (5,),
        )


@pytest.fixture
def two_dim_array_generator():
    return lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)


@pytest.fixture
def two_dim_virtual_array(numpy_like, two_dim_array_generator, shape_generator_param):
    if shape_generator_param is None:
        return VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=two_dim_array_generator,
        )
    else:
        return VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=two_dim_array_generator,
            shape_generator=lambda: (2, 3),
        )


@pytest.fixture
def scalar_array_generator():
    return lambda: np.array(42, dtype=np.int64)


@pytest.fixture
def scalar_virtual_array(numpy_like, scalar_array_generator, shape_generator_param):
    if shape_generator_param is None:
        return VirtualArray(
            numpy_like,
            shape=(),
            dtype=np.dtype(np.int64),
            generator=scalar_array_generator,
        )
    else:
        return VirtualArray(
            numpy_like,
            shape=(),
            dtype=np.dtype(np.int64),
            generator=scalar_array_generator,
            shape_generator=lambda: (),
        )


@pytest.fixture
def float_array_generator():
    return lambda: np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)


@pytest.fixture
def float_virtual_array(numpy_like, float_array_generator, shape_generator_param):
    if shape_generator_param is None:
        return VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
        )
    else:
        return VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
            shape_generator=lambda: (5,),
        )


# Test initialization
def test_init_valid(numpy_like, simple_array_generator, shape_generator_param):
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=simple_array_generator,
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=simple_array_generator,
            shape_generator=lambda: (5,),
        )

    assert va._shape == (unknown_length,)
    assert va.shape == (5,)

    assert va.dtype == np.dtype(np.int64)
    if shape_generator_param is None:
        assert va.is_materialized
    else:
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


def test_shape(virtual_array, two_dim_virtual_array, shape_generator_param):
    assert virtual_array._shape == (unknown_length,)
    assert two_dim_virtual_array._shape == (unknown_length, 3)
    assert virtual_array.shape == (5,)
    assert two_dim_virtual_array.shape == (2, 3)


def test_ndim(virtual_array, two_dim_virtual_array, scalar_virtual_array):
    assert virtual_array.ndim == 1
    assert two_dim_virtual_array.ndim == 2
    assert scalar_virtual_array.ndim == 0


def test_size(
    virtual_array, two_dim_virtual_array, scalar_virtual_array, shape_generator_param
):
    if shape_generator_param is None:
        # Without shape_generator, we know exact sizes
        assert virtual_array.size == 5
        assert two_dim_virtual_array.size == 6
    else:
        # With shape_generator, materializing to check size
        assert virtual_array.materialize().size == 5
        assert two_dim_virtual_array.materialize().size == 6

    assert scalar_virtual_array.size == 1


def test_nbytes_unmaterialized(virtual_array, shape_generator_param):
    if shape_generator_param is None:
        assert virtual_array.nbytes == unknown_length
    else:
        # Materialize to check bytes
        virtual_array.get_shape()
        assert virtual_array.nbytes == 5 * virtual_array.dtype.itemsize


def test_nbytes_materialized(virtual_array):
    virtual_array.materialize()
    assert virtual_array.nbytes == np.array([1, 2, 3, 4, 5], dtype=np.int64).nbytes


def test_strides(virtual_array, shape_generator_param):
    if shape_generator_param is not None:
        virtual_array.materialize()

    expected_strides = np.array([1, 2, 3, 4, 5], dtype=np.int64).strides
    assert virtual_array.strides == expected_strides


# Test materialization
def test_materialize(virtual_array):
    result = virtual_array.materialize()
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))
    assert virtual_array.is_materialized

    # After materialization, shape should be concrete regardless of shape_generator
    assert virtual_array.shape == (5,)


def test_is_materialized(virtual_array):
    assert not virtual_array.is_materialized
    virtual_array.materialize()
    assert virtual_array.is_materialized


def test_materialize_shape(numpy_like):
    # Generator returns array with different shape than declared
    va = VirtualArray(
        numpy_like,
        shape=(unknown_length,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3], dtype=np.int64),
    )
    va.get_shape()
    assert va.shape == (3,)


def test_materialize_shape_mismatch(numpy_like):
    # Generator returns array with different shape than declared
    va = VirtualArray(
        numpy_like,
        shape=(unknown_length,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3], dtype=np.int64),
        shape_generator=lambda: (5,),
    )
    va.get_shape()
    with pytest.raises(
        ValueError,
        match=r"had shape \(5,\) before materialization while the materialized array has shape \(3,\)",
    ):
        va.materialize()


def test_materialize_dtype_mismatch(numpy_like):
    # Generator returns array with different dtype than declared
    with pytest.raises(
        ValueError,
        match=r"had dtype int64 before materialization while the materialized array has dtype float64",
    ):
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=lambda: np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        va.materialize()


# Test transpose
def test_T_unmaterialized(two_dim_virtual_array, shape_generator_param):
    transposed = two_dim_virtual_array.T
    assert isinstance(transposed, VirtualArray)

    assert transposed._shape == (3, unknown_length)
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
        shape=(unknown_length,),
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


# Test shape_generator property
def test_shape_generator(virtual_array, shape_generator_param):
    if shape_generator_param is None:
        assert virtual_array._shape == (unknown_length,)
        assert virtual_array._shape_generator is None
        virtual_array.get_shape()
        assert virtual_array.is_materialized
        assert virtual_array._shape == (5,)
    else:
        assert virtual_array._shape == (unknown_length,)
        assert virtual_array._shape_generator is not None
        # Can't directly compare lambdas, so check if it's callable
        assert callable(virtual_array._shape_generator)
        virtual_array.get_shape()
        assert virtual_array._shape == (5,)
        assert not virtual_array.is_materialized
        with pytest.raises(
            AssertionError, match="this shape_generator should never be run!"
        ):
            assert virtual_array._shape_generator() == (5,)


# Test copy
def test_copy(virtual_array, shape_generator_param):
    copy = virtual_array.copy()
    assert isinstance(copy, VirtualArray)
    assert copy._shape == virtual_array._shape
    assert copy.shape == virtual_array.shape
    assert copy.dtype == virtual_array.dtype
    if shape_generator_param is None:
        assert copy.is_materialized
    assert id(copy) != id(virtual_array)  # Different objects
    assert copy._generator is virtual_array._generator


# Test tolist
def test_tolist(virtual_array):
    assert virtual_array.tolist() == [1, 2, 3, 4, 5]
    # Should be materialized after this
    assert virtual_array.is_materialized


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
    # Should be materialized after this
    assert virtual_array.is_materialized


# Test data
def test_data(virtual_array):
    data = virtual_array.data
    assert data is not None
    # Should be materialized after this
    assert virtual_array.is_materialized


# Test __repr__ and __str__
def test_repr(virtual_array, shape_generator_param):
    repr_str = repr(virtual_array)
    assert "VirtualArray" in repr_str
    assert "shape=(awkward._nplikes.shape.unknown_length,)" in repr_str


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
    # Should be materialized after this
    assert virtual_array.is_materialized


def test_getitem_slice(virtual_array, shape_generator_param):
    sliced = virtual_array[1:4]
    assert isinstance(sliced, VirtualArray)

    if shape_generator_param is None:
        assert sliced.shape == (3,)
    else:
        # With shape_generator, need to materialize to check shape
        materialized = sliced.materialize()
        assert materialized.shape == (3,)

    np.testing.assert_array_equal(sliced.materialize(), np.array([2, 3, 4]))


def test_getitem_slice_with_step(virtual_array, shape_generator_param):
    sliced = virtual_array[::2]
    assert isinstance(sliced, VirtualArray)

    if shape_generator_param is None:
        assert sliced.shape == (3,)
    else:
        # With shape_generator, need to materialize to check shape
        materialized = sliced.materialize()
        assert materialized.shape == (3,)

    np.testing.assert_array_equal(sliced.materialize(), np.array([1, 3, 5]))


def test_getitem_slice_with_unknown_length():
    nplike = Numpy.instance()
    va = VirtualArray(
        nplike,
        shape=(unknown_length,),
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
def test_len(virtual_array, two_dim_virtual_array, shape_generator_param):
    if shape_generator_param is None:
        assert len(virtual_array) == 5
        assert len(two_dim_virtual_array) == 2
    else:
        # Materialize to get concrete length
        virtual_array.materialize()
        two_dim_virtual_array.materialize()
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
    # Should be materialized after this
    assert virtual_array.is_materialized


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
    nplike = Numpy.instance()
    va1 = VirtualArray(
        nplike,
        shape=(unknown_length,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3], dtype=np.int64),
    )
    va2 = VirtualArray(
        nplike,
        shape=(unknown_length,),
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
def test_float_array_init(numpy_like, float_array_generator, shape_generator_param):
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.float64,
            generator=float_array_generator,
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.float64,
            generator=float_array_generator,
            shape_generator=lambda: (5,),
        )

    assert va._shape == (unknown_length,)
    assert va.shape == (5,)

    assert va.dtype == np.dtype(np.float64)
    if shape_generator_param is not None:
        assert not va.is_materialized
    else:
        assert va.is_materialized


def test_float_array_materialize(float_virtual_array):
    result = float_virtual_array.materialize()
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert float_virtual_array.is_materialized
    assert float_virtual_array.shape == (
        5,
    )  # Now concrete regardless of shape_generator


def test_float_array_slicing(numpy_like, float_array_generator, shape_generator_param):
    # Test basic slice
    if shape_generator_param is None:
        float_virtual_array1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
        )
    else:
        float_virtual_array1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
            shape_generator=lambda: (5,),
        )

    sliced = float_virtual_array1[1:4]
    assert isinstance(sliced, VirtualArray)

    if shape_generator_param is None:
        assert sliced.shape == (3,)
    else:
        # Materialize to check shape
        result = sliced.materialize()
        assert result.shape == (3,)

    np.testing.assert_array_almost_equal(
        sliced.materialize(), np.array([2.2, 3.3, 4.4])
    )

    # Test step slice
    if shape_generator_param is None:
        float_virtual_array2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
        )
    else:
        float_virtual_array2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
            shape_generator=lambda: (5,),
        )

    sliced_step = float_virtual_array2[::2]
    assert isinstance(sliced_step, VirtualArray)

    if shape_generator_param is None:
        assert sliced_step.shape == (3,)
    else:
        # Materialize to check shape
        result = sliced_step.materialize()
        assert result.shape == (3,)

    np.testing.assert_array_almost_equal(
        sliced_step.materialize(), np.array([1.1, 3.3, 5.5])
    )

    # Test negative step
    if shape_generator_param is None:
        float_virtual_array3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
        )
    else:
        float_virtual_array3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
            shape_generator=lambda: (5,),
        )

    sliced_neg = float_virtual_array3[::-1]
    assert isinstance(sliced_neg, VirtualArray)

    if shape_generator_param is None:
        assert sliced_neg.shape == (5,)
    else:
        # Materialize to check shape
        result = sliced_neg.materialize()
        assert result.shape == (5,)

    np.testing.assert_array_almost_equal(
        sliced_neg.materialize(), np.array([5.5, 4.4, 3.3, 2.2, 1.1])
    )

    # Test complex slice
    if shape_generator_param is None:
        float_virtual_array4 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
        )
    else:
        float_virtual_array4 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=float_array_generator,
            shape_generator=lambda: (5,),
        )

    sliced_complex = float_virtual_array4[4:1:-2]
    assert isinstance(sliced_complex, VirtualArray)

    if shape_generator_param is None:
        assert sliced_complex.shape == (2,)
    else:
        # Materialize to check shape
        result = sliced_complex.materialize()
        assert result.shape == (2,)

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
    assert float_virtual_array.shape == virtual_array.shape  # After materialization
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
def test_float_array_rounding(shape_generator_param):
    nplike = Numpy.instance()

    if shape_generator_param is None:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.float64,
            generator=lambda: np.array([1.1, 2.5, 3.7, 4.2, 5.9], dtype=np.float64),
        )
    else:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.float64,
            generator=lambda: np.array([1.1, 2.5, 3.7, 4.2, 5.9], dtype=np.float64),
            shape_generator=lambda: (5,),
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
def test_float_array_nan(shape_generator_param):
    nplike = Numpy.instance()

    if shape_generator_param is None:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.float64,
            generator=lambda: np.array(
                [1.1, np.nan, 3.3, np.nan, 5.5], dtype=np.float64
            ),
        )
    else:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.float64,
            generator=lambda: np.array(
                [1.1, np.nan, 3.3, np.nan, 5.5], dtype=np.float64
            ),
            shape_generator=lambda: (5,),
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
        shape=(unknown_length, 3),
        dtype=np.int64,
        generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
    )

    # Test advanced indexing
    sliced = va[:, 1]
    # This should materialize because it's not a simple first-dimension slice
    assert isinstance(sliced, np.ndarray)
    np.testing.assert_array_equal(sliced, np.array([2, 5]))


# Test empty array handling
def test_empty_array(shape_generator_param):
    nplike = Numpy.instance()

    if shape_generator_param is None:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=lambda: np.array([], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=lambda: np.array([], dtype=np.int64),
            shape_generator=lambda: (0,),
        )

    materialized = va.materialize()
    assert len(materialized) == 0
    assert va.size == 0

    if shape_generator_param is None:
        assert va.shape == (0,)
    else:
        # After materialization, shape should be concrete
        assert va.shape == (0,)


# Test with structured dtypes
def test_structured_dtype(shape_generator_param):
    dtype = np.dtype([("name", "U10"), ("age", "i4"), ("weight", "f8")])
    data = np.array(
        [("Alice", 25, 55.0), ("Bob", 30, 70.5), ("Charlie", 35, 65.2)], dtype=dtype
    )

    nplike = Numpy.instance()

    if shape_generator_param is None:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=dtype,
            generator=lambda: data,
        )
    else:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=dtype,
            generator=lambda: data,
            shape_generator=lambda: (3,),
        )

    assert va.dtype == dtype
    materialized = va.materialize()
    assert materialized["name"][1] == "Bob"
    assert materialized["age"][2] == 35
    assert materialized["weight"][0] == 55.0


# Test with large arrays to check memory efficiency
def test_large_array_memory(shape_generator_param):
    # Create a large array that would consume significant memory
    nplike = Numpy.instance()

    # Define a generator that would create a large array
    def large_array_generator():
        return np.ones((1000, 1000), dtype=np.float64)

    if shape_generator_param is None:
        va = VirtualArray(
            nplike,
            shape=(unknown_length, 1000),
            dtype=np.float64,
            generator=large_array_generator,
        )
    else:
        va = VirtualArray(
            nplike,
            shape=(unknown_length, 1000),
            dtype=np.float64,
            generator=large_array_generator,
            shape_generator=lambda: (1000, 1000),
        )

    assert va.nbytes == unknown_length
    va.get_shape()
    assert va.nbytes == 1000 * 1000 * 8
    if shape_generator_param is not None:
        assert not va.is_materialized

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
        shape=(unknown_length,),
        dtype=np.int64,
        generator=failing_generator,
    )

    with pytest.raises(ValueError, match="Generator failure test"):
        va.get_shape()


# Test nested VirtualArrays (generator returns another VirtualArray)
def test_nested_virtual_arrays(shape_generator_param):
    nplike = Numpy.instance()

    # Inner virtual array
    if shape_generator_param is None:
        inner_va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=lambda: np.array([10, 20, 30], np.int64),
        )
    else:
        inner_va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=lambda: np.array([10, 20, 30], np.int64),
            shape_generator=lambda: (3,),
        )

    # Outer virtual array, generator returns inner virtual array
    if shape_generator_param is None:
        outer_va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=lambda: inner_va,
        )
    else:
        outer_va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.int64,
            generator=lambda: inner_va,
            shape_generator=lambda: (3,),
        )

    # Should materialize both
    result = outer_va.materialize()
    np.testing.assert_array_equal(result, np.array([10, 20, 30]))
    assert inner_va.is_materialized
    assert outer_va.is_materialized


# Test with complex numbers
def test_complex_numbers(shape_generator_param):
    nplike = Numpy.instance()

    if shape_generator_param is None:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.complex128,
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
        )
    else:
        va = VirtualArray(
            nplike,
            shape=(unknown_length,),
            dtype=np.complex128,
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
            shape_generator=lambda: (3,),
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
        shape=(unknown_length,),
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
            shape=(unknown_length,),
            dtype=np.int64,
            generator=create_generator(array_length),
        )

        sliced = va[slice_obj]

        if shape_generator_param is None:
            # Without shape_generator, we can check the shape directly
            assert isinstance(sliced, VirtualArray)
            assert sliced.shape[0] == expected_length, f"Failed for slice {slice_obj}"
        else:
            # With shape_generator, need to materialize to check shape
            result = sliced.materialize()
            assert len(result) == expected_length, f"Failed for slice {slice_obj}"


# Test nplike of obj
def test_nplike_of_obj(
    virtual_array, float_virtual_array, numpy_like, shape_generator_param
):
    assert nplike_of_obj(virtual_array) is numpy_like
    assert nplike_of_obj(float_virtual_array) is numpy_like


# Test backend of obj
def test_backend_of_obj(virtual_array, float_virtual_array, shape_generator_param):
    assert backend_of_obj(virtual_array).name == "cpu"
    assert backend_of_obj(float_virtual_array).name == "cpu"


# Test array creation methods with VirtualArray
def test_asarray_virtual_array_unmaterialized(
    numpy_like, virtual_array, shape_generator_param
):
    # Test with unmaterialized VirtualArray
    result = numpy_like.asarray(virtual_array)
    assert result is virtual_array  # Should return the same object
    assert not virtual_array.is_materialized


def test_asarray_virtual_array_materialized(
    numpy_like, virtual_array, shape_generator_param
):
    # Test with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.asarray(virtual_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_asarray_virtual_array_with_dtype(
    numpy_like, virtual_array, shape_generator_param
):
    # Test with dtype parameter
    out = numpy_like.asarray(virtual_array, dtype=np.float64)
    assert isinstance(out, VirtualArray)
    assert out.dtype == np.dtype(np.float64)
    assert not out.is_materialized
    assert out.materialize().dtype == np.dtype(np.float64)


def test_asarray_virtual_array_with_copy(
    numpy_like, virtual_array, shape_generator_param
):
    # Test with copy parameter
    virtual_array.materialize()
    with pytest.raises(
        ValueError,
        match="asarray was called with copy=False for an array of a different dtype",
    ):
        # Should raise because we're trying to change the dtype without copying
        numpy_like.asarray(virtual_array, dtype=np.float64, copy=False)


def test_ascontiguousarray_unmaterialized(
    numpy_like, virtual_array, shape_generator_param
):
    # Test with unmaterialized VirtualArray
    result = numpy_like.ascontiguousarray(virtual_array)
    assert isinstance(result, VirtualArray)
    assert not result.is_materialized
    assert result.shape == virtual_array.shape
    assert result.dtype == virtual_array.dtype


def test_ascontiguousarray_materialized(
    numpy_like, virtual_array, shape_generator_param
):
    # Test with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.ascontiguousarray(virtual_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_frombuffer_with_virtual_array(
    numpy_like, virtual_array, shape_generator_param
):
    # Test frombuffer with VirtualArray (should raise TypeError)
    with pytest.raises(
        TypeError, match="virtual arrays are not supported in `frombuffer`"
    ):
        numpy_like.frombuffer(virtual_array)


# Test array creation methods using materialization info
def test_zeros_like_unmaterialized(numpy_like, virtual_array, shape_generator_param):
    # Test zeros_like with unmaterialized VirtualArray
    result = numpy_like.zeros_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.zeros(5, dtype=np.int64))
    if shape_generator_param is None:
        assert virtual_array.is_materialized
    else:
        assert not virtual_array.is_materialized


def test_zeros_like_materialized(numpy_like, virtual_array, shape_generator_param):
    # Test zeros_like with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.zeros_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.zeros(5, dtype=np.int64))


def test_ones_like_unmaterialized(numpy_like, virtual_array, shape_generator_param):
    # Test ones_like with unmaterialized VirtualArray
    result = numpy_like.ones_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.ones(5, dtype=np.int64))
    if shape_generator_param is None:
        assert virtual_array.is_materialized
    else:
        assert not virtual_array.is_materialized


def test_ones_like_materialized(numpy_like, virtual_array, shape_generator_param):
    # Test ones_like with materialized VirtualArray
    virtual_array.materialize()
    result = numpy_like.ones_like(virtual_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.ones(5, dtype=np.int64))


def test_full_like_unmaterialized(numpy_like, virtual_array, shape_generator_param):
    # Test full_like with unmaterialized VirtualArray
    result = numpy_like.full_like(virtual_array, 7)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.full(5, 7, dtype=np.int64))
    if shape_generator_param is None:
        assert virtual_array.is_materialized
    else:
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
def test_array_equal_with_virtual_arrays(
    numpy_like, virtual_array, shape_generator_param
):
    # Create two identical VirtualArrays
    va1 = virtual_array
    if shape_generator_param is None:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    else:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
            shape_generator=lambda: (5,),
        )

    # Test array_equal
    result = numpy_like.array_equal(va1, va2)
    assert result is True

    # Test with a different VirtualArray
    if shape_generator_param is None:
        va3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array(
                [1, 2, 3, 4, 6], dtype=np.int64
            ),  # Different last value
        )
    else:
        va3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array(
                [1, 2, 3, 4, 6], dtype=np.int64
            ),  # Different last value
            shape_generator=lambda: (5,),
        )

    result = numpy_like.array_equal(va1, va3)
    assert result is False


def test_array_equal_with_equal_nan(numpy_like, shape_generator_param):
    # Test array_equal with equal_nan=True
    if shape_generator_param is None:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, np.nan, 3.0], dtype=np.float64),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, np.nan, 3.0], dtype=np.float64),
        )
    else:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, np.nan, 3.0], dtype=np.float64),
            shape_generator=lambda: (3,),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, np.nan, 3.0], dtype=np.float64),
            shape_generator=lambda: (3,),
        )

    # Should be False by default (NaN != NaN)
    result = numpy_like.array_equal(va1, va2)
    assert bool(result) is False

    # Should be True with equal_nan=True
    result = numpy_like.array_equal(va1, va2, equal_nan=True)
    assert bool(result) is True


def test_searchsorted_with_virtual_arrays(
    numpy_like, virtual_array, shape_generator_param
):
    # Test searchsorted with VirtualArray
    if shape_generator_param is None:
        values = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, 3, 6], dtype=np.int64),
        )
    else:
        values = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([0, 3, 6], dtype=np.int64),
            shape_generator=lambda: (3,),
        )

    result = numpy_like.searchsorted(virtual_array, values)
    np.testing.assert_array_equal(
        result, np.array([0, 2, 5])
    )  # Indices where values would be inserted


# Test ufunc application with VirtualArray
def test_apply_ufunc_with_virtual_arrays(
    numpy_like, virtual_array, shape_generator_param
):
    # Test apply_ufunc with add operation
    result = numpy_like.apply_ufunc(np.add, "__call__", [virtual_array, 10])
    np.testing.assert_array_equal(result, np.array([11, 12, 13, 14, 15]))

    # Test apply_ufunc with multiple VirtualArrays
    if shape_generator_param is None:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
        )
    else:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
            shape_generator=lambda: (5,),
        )

    result = numpy_like.apply_ufunc(np.multiply, "__call__", [virtual_array, va2])
    np.testing.assert_array_equal(result, np.array([10, 40, 90, 160, 250]))


# Test manipulation functions with VirtualArray
def test_broadcast_arrays_with_virtual_arrays(
    numpy_like, virtual_array, shape_generator_param
):
    # Test broadcast_arrays with VirtualArrays
    if shape_generator_param is None:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10], dtype=np.int64),
        )
    else:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10], dtype=np.int64),
            shape_generator=lambda: (1,),
        )

    result = numpy_like.broadcast_arrays(virtual_array, va2)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(result[1], np.array([10, 10, 10, 10, 10]))


def test_reshape_unmaterialized(numpy_like, virtual_array, shape_generator_param):
    # Test reshape with unmaterialized VirtualArray
    result = numpy_like.reshape(virtual_array, (5, 1))
    assert isinstance(result, VirtualArray)
    assert not result.is_materialized
    assert result.shape == (5, 1)

    # Test reshape with -1 dimension
    result = numpy_like.reshape(virtual_array, (-1, 1))
    if shape_generator_param is None:
        assert isinstance(result, np.ndarray)
        assert virtual_array.is_materialized
    else:
        assert isinstance(result, VirtualArray)
        assert not virtual_array.is_materialized
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


def test_where_with_virtual_arrays(numpy_like, virtual_array, shape_generator_param):
    # Test where with VirtualArrays
    if shape_generator_param is None:
        condition = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array(
                [True, False, True, False, True], dtype=np.bool_
            ),
        )
        x1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
        )
    else:
        condition = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array(
                [True, False, True, False, True], dtype=np.bool_
            ),
            shape_generator=lambda: (5,),
        )
        x1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
            shape_generator=lambda: (5,),
        )

    result = numpy_like.where(condition, virtual_array, x1)
    np.testing.assert_array_equal(result, np.array([1, 20, 3, 40, 5]))


def test_unique_values_with_virtual_array(numpy_like, shape_generator_param):
    # Test unique_values with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 2, 3, 3, 3, 1], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 2, 3, 3, 3, 1], dtype=np.int64),
            shape_generator=lambda: (7,),
        )

    result = numpy_like.unique_values(va)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_unique_all_with_virtual_array(numpy_like, shape_generator_param):
    # Test unique_all with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 2, 3, 3, 3, 1], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 2, 3, 3, 3, 1], dtype=np.int64),
            shape_generator=lambda: (7,),
        )

    result = numpy_like.unique_all(va)
    np.testing.assert_array_equal(result.values, np.array([1, 2, 3]))
    np.testing.assert_array_equal(result.counts, np.array([2, 2, 3]))
    # Check inverse indices have original shape
    assert result.inverse_indices.shape == (7,)


def test_sort_with_virtual_array(numpy_like, shape_generator_param):
    # Test sort with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
            shape_generator=lambda: (5,),
        )

    # Sort ascending
    result = numpy_like.sort(va)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))

    # Sort descending
    result = numpy_like.sort(va, descending=True)
    np.testing.assert_array_equal(result, np.array([5, 4, 3, 2, 1]))

    # Test with 2D array and axis
    if shape_generator_param is None:
        va2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
        )
    else:
        va2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
            shape_generator=lambda: (2, 3),
        )

    result = numpy_like.sort(va2d, axis=1)
    np.testing.assert_array_equal(result, np.array([[1, 2, 3], [4, 5, 6]]))


def test_concat_with_virtual_arrays(numpy_like, virtual_array, shape_generator_param):
    # Test concat with VirtualArrays
    if shape_generator_param is None:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([6, 7, 8], dtype=np.int64),
        )
    else:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([6, 7, 8], dtype=np.int64),
            shape_generator=lambda: (3,),
        )

    result = numpy_like.concat([virtual_array, va2])
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    # Test with axis parameter
    if shape_generator_param is None:
        va2d1 = VirtualArray(
            numpy_like,
            shape=(unknown_length, 2),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 2], [3, 4]], dtype=np.int64),
        )
        va2d2 = VirtualArray(
            numpy_like,
            shape=(unknown_length, 2),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[5, 6], [7, 8]], dtype=np.int64),
        )
    else:
        va2d1 = VirtualArray(
            numpy_like,
            shape=(unknown_length, 2),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 2], [3, 4]], dtype=np.int64),
            shape_generator=lambda: (2, 2),
        )
        va2d2 = VirtualArray(
            numpy_like,
            shape=(unknown_length, 2),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[5, 6], [7, 8]], dtype=np.int64),
            shape_generator=lambda: (2, 2),
        )

    result = numpy_like.concat([va2d1, va2d2], axis=1)
    np.testing.assert_array_equal(result, np.array([[1, 2, 5, 6], [3, 4, 7, 8]]))


def test_repeat_with_virtual_array(numpy_like, virtual_array, shape_generator_param):
    # Test repeat with VirtualArray
    result = numpy_like.repeat(virtual_array, 2)
    np.testing.assert_array_equal(result, np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))

    # Test with axis parameter
    if shape_generator_param is None:
        va2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        )
    else:
        va2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            shape_generator=lambda: (2, 3),
        )

    result = numpy_like.repeat(va2d, 2, axis=0)
    np.testing.assert_array_equal(
        result, np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]])
    )


def test_stack_with_virtual_arrays(numpy_like, virtual_array, shape_generator_param):
    # Test stack with VirtualArrays
    if shape_generator_param is None:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([6, 7, 8, 9, 10], dtype=np.int64),
        )
    else:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([6, 7, 8, 9, 10], dtype=np.int64),
            shape_generator=lambda: (5,),
        )

    result = numpy_like.stack([virtual_array, va2])
    np.testing.assert_array_equal(result, np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))

    # Test with axis parameter
    result = numpy_like.stack([virtual_array, va2], axis=1)
    np.testing.assert_array_equal(
        result, np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]])
    )


def test_packbits_with_virtual_array(numpy_like, shape_generator_param):
    # Test packbits with VirtualArray of booleans
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool),
            shape_generator=lambda: (8,),
        )

    result = numpy_like.packbits(va)
    np.testing.assert_array_equal(
        result, np.array([170], dtype=np.uint8)
    )  # 10101010 in binary = 170


def test_unpackbits_with_virtual_array(numpy_like, shape_generator_param):
    # Test unpackbits with VirtualArray of uint8
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.uint8),
            generator=lambda: np.array([170], dtype=np.uint8),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.uint8),
            generator=lambda: np.array([170], dtype=np.uint8),
            shape_generator=lambda: (1,),
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
def test_add_with_virtual_arrays(numpy_like, virtual_array, shape_generator_param):
    # Test add with VirtualArrays
    if shape_generator_param is None:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
        )
    else:
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 20, 30, 40, 50], dtype=np.int64),
            shape_generator=lambda: (5,),
        )

    result = numpy_like.add(virtual_array, va2)
    np.testing.assert_array_equal(result, np.array([11, 22, 33, 44, 55]))


def test_logical_or_with_virtual_arrays(numpy_like, shape_generator_param):
    # Test logical_or with VirtualArrays
    if shape_generator_param is None:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, True, False, False], dtype=np.bool_),
        )
    else:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, True, False, False], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )

    result = numpy_like.logical_or(va1, va2)
    np.testing.assert_array_equal(result, np.array([True, True, True, False]))


def test_logical_and_with_virtual_arrays(numpy_like, shape_generator_param):
    # Test logical_and with VirtualArrays
    if shape_generator_param is None:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, True, False, False], dtype=np.bool_),
        )
    else:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, True, False, False], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )

    result = numpy_like.logical_and(va1, va2)
    np.testing.assert_array_equal(result, np.array([True, False, False, False]))


def test_logical_not_with_virtual_array(numpy_like, shape_generator_param):
    # Test logical_not with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, False], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )

    result = numpy_like.logical_not(va)
    np.testing.assert_array_equal(result, np.array([False, True, False, True]))


# Test mathematical operations
def test_sqrt_with_virtual_array(numpy_like, shape_generator_param):
    # Test sqrt with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float64),
            shape_generator=lambda: (4,),
        )

    result = numpy_like.sqrt(va)
    np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0, 4.0, 5.0]))


def test_exp_with_virtual_array(numpy_like, shape_generator_param):
    # Test exp with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([0.0, 1.0, 2.0], dtype=np.float64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([0.0, 1.0, 2.0], dtype=np.float64),
            shape_generator=lambda: (3,),
        )

    result = numpy_like.exp(va)
    np.testing.assert_array_almost_equal(result, np.array([1.0, np.e, np.e**2]))


def test_divide_with_virtual_arrays(numpy_like, shape_generator_param):
    # Test divide with VirtualArrays
    if shape_generator_param is None:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float64),
        )
    else:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
            shape_generator=lambda: (4,),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float64),
            shape_generator=lambda: (4,),
        )

    result = numpy_like.divide(va1, va2)
    np.testing.assert_array_almost_equal(result, np.array([5.0, 5.0, 6.0, 5.0]))


# Test special operations
def test_nan_to_num_with_virtual_array(numpy_like, shape_generator_param):
    # Test nan_to_num with VirtualArray containing NaN and infinity
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array(
                [1.0, np.nan, np.inf, -np.inf], dtype=np.float64
            ),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array(
                [1.0, np.nan, np.inf, -np.inf], dtype=np.float64
            ),
            shape_generator=lambda: (4,),
        )

    result = numpy_like.nan_to_num(va)
    # NaN becomes 0.0, inf becomes large finite number, -inf becomes large negative number
    assert result[0] == 1.0
    assert result[1] == 0.0
    assert result[2] > 1e300  # Large positive number
    assert result[3] < -1e300  # Large negative number


def test_isclose_with_virtual_arrays(numpy_like, shape_generator_param):
    # Test isclose with VirtualArrays
    if shape_generator_param is None:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0001, 2.0, 3.1, np.nan], dtype=np.float64),
        )
    else:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64),
            shape_generator=lambda: (4,),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0001, 2.0, 3.1, np.nan], dtype=np.float64),
            shape_generator=lambda: (4,),
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


def test_isnan_with_virtual_array(numpy_like, shape_generator_param):
    # Test isnan with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float64),
            shape_generator=lambda: (4,),
        )

    result = numpy_like.isnan(va)
    np.testing.assert_array_equal(result, np.array([False, True, False, True]))


# Test reduction operations
def test_all_with_virtual_array(numpy_like, shape_generator_param):
    # Test all with VirtualArray
    if shape_generator_param is None:
        va_all_true = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, True, True, True], dtype=np.bool_),
        )
        va_mixed = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, True], dtype=np.bool_),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array(
                [[True, True, False], [True, True, True]], dtype=np.bool_
            ),
        )
    else:
        va_all_true = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, True, True, True], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )
        va_mixed = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([True, False, True, True], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array(
                [[True, True, False], [True, True, True]], dtype=np.bool_
            ),
            shape_generator=lambda: (2, 3),
        )

    # Test all(all True)
    result = numpy_like.all(va_all_true)
    assert result

    # Test all(mixed)
    result = numpy_like.all(va_mixed)
    assert not result

    # Test with axis parameter
    result = numpy_like.all(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([False, True]))

    # Test with keepdims
    result = numpy_like.all(va_2d, axis=1, keepdims=True)
    np.testing.assert_array_equal(result, np.array([[False], [True]]))


def test_any_with_virtual_array(numpy_like, shape_generator_param):
    # Test any with VirtualArray
    if shape_generator_param is None:
        va_all_false = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([False, False, False, False], dtype=np.bool_),
        )
        va_mixed = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([False, False, True, False], dtype=np.bool_),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array(
                [[False, False, False], [False, True, False]], dtype=np.bool_
            ),
        )
    else:
        va_all_false = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([False, False, False, False], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )
        va_mixed = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array([False, False, True, False], dtype=np.bool_),
            shape_generator=lambda: (4,),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.bool_),
            generator=lambda: np.array(
                [[False, False, False], [False, True, False]], dtype=np.bool_
            ),
            shape_generator=lambda: (2, 3),
        )

    # Test any(all False)
    result = numpy_like.any(va_all_false)
    assert not result

    # Test any(mixed)
    result = numpy_like.any(va_mixed)
    assert result

    # Test with axis parameter
    result = numpy_like.any(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([False, True]))


def test_min_with_virtual_array(numpy_like, shape_generator_param):
    # Test min with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
            shape_generator=lambda: (5,),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
            shape_generator=lambda: (2, 3),
        )

    # Test overall min
    result = numpy_like.min(va)
    assert result == 1

    # Test with 2D array and axis
    result = numpy_like.min(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([1, 4]))


def test_max_with_virtual_array(numpy_like, shape_generator_param):
    # Test max with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([5, 3, 1, 4, 2], dtype=np.int64),
            shape_generator=lambda: (5,),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int64),
            shape_generator=lambda: (2, 3),
        )

    # Test overall max
    result = numpy_like.max(va)
    assert result == 5

    # Test with 2D array and axis
    result = numpy_like.max(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([3, 6]))


def test_count_nonzero_with_virtual_array(numpy_like, shape_generator_param):
    # Test count_nonzero with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 0, 3, 0, 5, 0], dtype=np.int64),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 0, 2], [0, 4, 0]], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 0, 3, 0, 5, 0], dtype=np.int64),
            shape_generator=lambda: (6,),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 0, 2], [0, 4, 0]], dtype=np.int64),
            shape_generator=lambda: (2, 3),
        )

    # Test overall count
    result = numpy_like.count_nonzero(va)
    assert result == 3

    # Test with 2D array and axis
    result = numpy_like.count_nonzero(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([2, 1]))


def test_cumsum_with_virtual_array(numpy_like, shape_generator_param):
    # Test cumsum with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
            shape_generator=lambda: (5,),
        )
        va_2d = VirtualArray(
            numpy_like,
            shape=(unknown_length, 3),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            shape_generator=lambda: (2, 3),
        )

    # Test cumsum
    result = numpy_like.cumsum(va)
    np.testing.assert_array_equal(result, np.array([1, 3, 6, 10, 15]))

    # Test with 2D array and axis
    result = numpy_like.cumsum(va_2d, axis=1)
    np.testing.assert_array_equal(result, np.array([[1, 3, 6], [4, 9, 15]]))


def test_real_imag_with_complex_virtual_array(numpy_like, shape_generator_param):
    # Test real and imag with complex VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128),
            shape_generator=lambda: (3,),
        )

    # Test real
    result = numpy_like.real(va)
    np.testing.assert_array_equal(result, np.array([1.0, 3.0, 5.0]))

    # Test imag
    result = numpy_like.imag(va)
    np.testing.assert_array_equal(result, np.array([2.0, 4.0, 6.0]))


def test_angle_with_complex_virtual_array(numpy_like, shape_generator_param):
    # Test angle with complex VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array(
                [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
            ),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.complex128),
            generator=lambda: np.array(
                [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128
            ),
            shape_generator=lambda: (4,),
        )

    # Test angle in radians
    result = numpy_like.angle(va)
    np.testing.assert_array_almost_equal(
        result, np.array([0, np.pi / 2, np.pi, -np.pi / 2])
    )

    # Test angle in degrees
    result = numpy_like.angle(va, deg=True)
    np.testing.assert_array_almost_equal(result, np.array([0, 90, 180, -90]))


def test_round_with_virtual_array(numpy_like, shape_generator_param):
    # Test round with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.234, 2.567, 3.499, 4.501], dtype=np.float64),
            shape_generator=lambda: (4,),
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
def test_materialize_if_virtual_function(numpy_like, shape_generator_param):
    # Test the materialize_if_virtual utility function directly

    # Create a mix of VirtualArrays and regular arrays
    if shape_generator_param is None:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3], dtype=np.int64),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([4, 5, 6], dtype=np.int64),
        )
    else:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3], dtype=np.int64),
            shape_generator=lambda: (3,),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([4, 5, 6], dtype=np.int64),
            shape_generator=lambda: (3,),
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
    if shape_generator_param is None:
        va3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 11], dtype=np.int64),
        )
    else:
        va3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([10, 11], dtype=np.int64),
            shape_generator=lambda: (2,),
        )

    va3.materialize()  # Pre-materialize

    results = materialize_if_virtual(va3, regular_array)
    assert len(results) == 2
    np.testing.assert_array_equal(results[0], np.array([10, 11]))
    np.testing.assert_array_equal(results[1], np.array([7, 8, 9]))


def test_operations_with_multiple_virtual_arrays(numpy_like, shape_generator_param):
    # Test a complex operation involving multiple VirtualArrays
    if shape_generator_param is None:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([4.0, 5.0, 6.0], dtype=np.float64),
        )
        va3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([7.0, 8.0, 9.0], dtype=np.float64),
        )
    else:
        va1 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([1.0, 2.0, 3.0], dtype=np.float64),
            shape_generator=lambda: (3,),
        )
        va2 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([4.0, 5.0, 6.0], dtype=np.float64),
            shape_generator=lambda: (3,),
        )
        va3 = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.float64),
            generator=lambda: np.array([7.0, 8.0, 9.0], dtype=np.float64),
            shape_generator=lambda: (3,),
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


def test_is_own_array_with_virtual_array(numpy_like, shape_generator_param):
    # Test is_own_array method with VirtualArray
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3], dtype=np.int64),
            shape_generator=lambda: (3,),
        )

    # Before materialization
    result = numpy_like.is_own_array(va)
    assert result  # Should be True because VirtualArray.nplike.ndarray is numpy.ndarray

    # After materialization
    va.materialize()
    result = numpy_like.is_own_array(va)
    assert result


def test_virtual_array_with_structured_dtype(numpy_like, shape_generator_param):
    # Test VirtualArray with structured dtype
    dtype = np.dtype([("name", "U10"), ("age", "i4"), ("weight", "f8")])

    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=dtype,
            generator=lambda: np.array(
                [("Alice", 25, 55.0), ("Bob", 30, 70.5)], dtype=dtype
            ),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=dtype,
            generator=lambda: np.array(
                [("Alice", 25, 55.0), ("Bob", 30, 70.5)], dtype=dtype
            ),
            shape_generator=lambda: (2,),
        )

    # Test properties
    assert va.dtype == dtype
    assert va._shape[0] == unknown_length
    assert va.shape[0] == 2

    # Test materialization
    materialized = va.materialize()
    assert materialized["name"][0] == "Alice"
    assert materialized["age"][1] == 30
    assert materialized["weight"][1] == 70.5


def test_virtual_array_with_empty_array(numpy_like, shape_generator_param):
    # Test VirtualArray with empty array
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([], dtype=np.int64),
            shape_generator=lambda: (0,),
        )

    # Test materialization
    materialized = va.materialize()
    assert len(materialized) == 0
    assert materialized.size == 0


def test_chained_operations_materialization(numpy_like, shape_generator_param):
    # Test that chained operations correctly materialize VirtualArrays
    if shape_generator_param is None:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
        )
    else:
        va = VirtualArray(
            numpy_like,
            shape=(unknown_length,),
            dtype=np.dtype(np.int64),
            generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
            shape_generator=lambda: (5,),
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
