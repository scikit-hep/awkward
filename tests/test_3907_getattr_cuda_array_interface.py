from __future__ import annotations

import pytest

import awkward as ak


def test_numpy_array_getattr_cuda_array_interface_raises_on_numpy_backend():
    """__getattr__ raises AttributeError for __cuda_array_interface__ on CPU backend."""

    layout = ak.Array([1.0, 2.0, 3.0]).layout

    with pytest.raises(AttributeError, match="not backed by a CuPy array"):
        _ = layout.__cuda_array_interface__

    # hasattr must also return False — not just raise
    assert not hasattr(layout, "__cuda_array_interface__")


def test_numpy_array_getattr_unknown_attribute_raises():
    """__getattr__ raises AttributeError for unknown attributes."""

    layout = ak.Array([1.0, 2.0, 3.0]).layout

    with pytest.raises(
        AttributeError, match="'NumpyArray' object has no attribute 'nonexistent'"
    ):
        _ = layout.nonexistent
