# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_from_numpy_attrs():
    """Test that ak.from_numpy correctly passes attrs parameter."""
    array = np.array([1, 2, 3])
    result = ak.from_numpy(array, attrs={"hello": "world"})
    assert result.attrs == {"hello": "world"}
    assert result.to_list() == [1, 2, 3]


def test_from_numpy_attrs_none():
    """Test that ak.from_numpy works with attrs=None."""
    array = np.array([1, 2, 3])
    result = ak.from_numpy(array, attrs=None)
    assert result.attrs == {}
    assert result.to_list() == [1, 2, 3]


def test_from_numpy_attrs_with_structured_array():
    """Test that ak.from_numpy passes attrs with structured arrays."""
    array = np.array([(1, 2.5), (3, 4.5)], dtype=[("x", "i4"), ("y", "f8")])
    result = ak.from_numpy(array, attrs={"source": "numpy"})
    assert result.attrs == {"source": "numpy"}
    assert ak.fields(result) == ["x", "y"]


def test_from_numpy_attrs_with_behavior():
    """Test that ak.from_numpy works with both attrs and behavior."""
    array = np.array([1, 2, 3])
    result = ak.from_numpy(
        array, attrs={"hello": "world"}, behavior={"custom": "behavior"}
    )
    assert result.attrs == {"hello": "world"}
    assert result.behavior == {"custom": "behavior"}


def test_from_cupy_attrs():
    """Test that ak.from_cupy correctly passes attrs parameter."""
    cupy = pytest.importorskip("cupy")
    array = cupy.array([1, 2, 3])
    result = ak.from_cupy(array, attrs={"foo": "bar"})
    assert result.attrs == {"foo": "bar"}
    assert result.to_list() == [1, 2, 3]


def test_from_cupy_attrs_none():
    """Test that ak.from_cupy works with attrs=None."""
    cupy = pytest.importorskip("cupy")
    array = cupy.array([1, 2, 3])
    result = ak.from_cupy(array, attrs=None)
    assert result.attrs == {}
    assert result.to_list() == [1, 2, 3]


def test_from_jax_attrs():
    """Test that ak.from_jax correctly passes attrs parameter."""
    jax = pytest.importorskip("jax")
    array = jax.numpy.array([1, 2, 3])
    result = ak.from_jax(array, attrs={"jax": "test"})
    assert result.attrs == {"jax": "test"}
    assert result.to_list() == [1, 2, 3]


def test_from_jax_attrs_none():
    """Test that ak.from_jax works with attrs=None."""
    jax = pytest.importorskip("jax")
    array = jax.numpy.array([1, 2, 3])
    result = ak.from_jax(array, attrs=None)
    assert result.attrs == {}
    assert result.to_list() == [1, 2, 3]


def test_from_dlpack_attrs():
    """Test that ak.from_dlpack correctly passes attrs parameter."""
    # DLPack is supported by NumPy 1.22+
    array = np.array([1, 2, 3])
    if not hasattr(array, "__dlpack__"):
        pytest.skip("NumPy version doesn't support DLPack")

    result = ak.from_dlpack(array, attrs={"dlpack": "test"})
    assert result.attrs == {"dlpack": "test"}
    assert result.to_list() == [1, 2, 3]


def test_from_dlpack_attrs_none():
    """Test that ak.from_dlpack works with attrs=None."""
    array = np.array([1, 2, 3])
    if not hasattr(array, "__dlpack__"):
        pytest.skip("NumPy version doesn't support DLPack")

    result = ak.from_dlpack(array, attrs=None)
    assert result.attrs == {}
    assert result.to_list() == [1, 2, 3]


def test_from_numpy_attrs_multidimensional():
    """Test that attrs work with multidimensional arrays."""
    array = np.array([[1, 2], [3, 4]])
    result = ak.from_numpy(array, regulararray=True, attrs={"shape": "2x2"})
    assert result.attrs == {"shape": "2x2"}
    assert result.to_list() == [[1, 2], [3, 4]]


def test_from_numpy_attrs_complex():
    """Test that attrs work with complex nested attributes."""
    array = np.array([1, 2, 3])
    complex_attrs = {"meta": {"version": "1.0", "source": "test"}, "id": 42}
    result = ak.from_numpy(array, attrs=complex_attrs)
    assert result.attrs == complex_attrs
    assert result.to_list() == [1, 2, 3]


def test_from_functions_attrs_preserved_in_operations():
    """Test that attrs from from_* functions are preserved in operations."""
    array = np.array([1, 2, 3])
    result = ak.from_numpy(array, attrs={"original": "data"})

    # Test that attrs survive slicing
    sliced = result[1:]
    assert sliced.attrs == {"original": "data"}
    assert sliced.to_list() == [2, 3]
