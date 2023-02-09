# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.typetracer import MaybeNone, is_unknown_scalar


def test():
    array = ak.Array([1, 2, 3, 4])
    typetracer_array = ak.Array(array.layout.to_typetracer(forget_length=True))
    typetracer_result = np.sqrt(typetracer_array)

    assert typetracer_result.type == ak.types.ArrayType(
        ak.types.NumpyType("float64"), unknown_length
    )


def test_add():
    left = ak.Array([1, 2, 3, 4])
    right = ak.Array([1, 2, 3, 4.0])
    typetracer_left = ak.Array(left.layout.to_typetracer(forget_length=True))
    typetracer_right = ak.Array(right.layout.to_typetracer(forget_length=True))
    typetracer_result = np.add(typetracer_left, typetracer_right)

    assert typetracer_result.type == ak.types.ArrayType(
        ak.types.NumpyType("float64"), unknown_length
    )


def test_add_scalar():
    array = ak.Array([1, 2, 3, 4])
    typetracer_array = ak.Array(array.layout.to_typetracer(forget_length=True))
    other = ak.min(typetracer_array, mask_identity=False, initial=10)
    assert is_unknown_scalar(other)

    typetracer_result = np.add(typetracer_array, other)
    assert typetracer_result.type == ak.types.ArrayType(
        ak.types.NumpyType("int64"), unknown_length
    )


def test_add_none_scalar():
    array = ak.Array([1, 2, 3, 4])
    typetracer_array = ak.Array(array.layout.to_typetracer(forget_length=True))
    other = ak.min(typetracer_array, mask_identity=True, initial=10)
    assert isinstance(other, MaybeNone)
    assert is_unknown_scalar(other.content)

    typetracer_result = np.add(typetracer_array, other)
    assert typetracer_result.type == ak.types.ArrayType(
        ak.types.NumpyType("int64"), unknown_length
    )
