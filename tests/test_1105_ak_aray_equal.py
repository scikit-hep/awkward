# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward.contents import NumpyArray
from awkward.index import Index64


def test_array_equal_simple():
    assert ak.array_equal(
        ak.Array([[1, 2], [], [3, 4, 5]]),
        ak.Array([[1, 2], [], [3, 4, 5]]),
    )


def test_array_equal_mixed_dtype():
    assert not ak.array_equal(
        ak.Array(np.array([1.5, 2.0, 3.25], dtype=np.float32)),
        ak.Array(np.array([1.5, 2.0, 3.25], dtype=np.float64)),
    )
    assert ak.array_equal(
        ak.Array(np.array([1.5, 2.0, 3.25], dtype=np.float32)),
        ak.Array(np.array([1.5, 2.0, 3.25], dtype=np.float64)),
        dtype_exact=False,
    )


def test_array_equal_on_listoffsets():
    a1 = ak.contents.ListOffsetArray(
        Index64(np.array([0, 3, 3, 5])), NumpyArray(np.arange(5) * 1.5)
    )
    a2 = ak.contents.ListOffsetArray(
        Index64(np.array([0, 3, 3, 5])),
        NumpyArray(np.arange(10) * 1.5),  # Longer array content than a1
    )
    assert ak.array_equal(a1, a2)

    # double jagged array
    a = ak.Array([[[1], [2, 3]], [[4, 5], [6]]])
    assert ak.array_equal(a, a)

    # different index same content
    a = ak.Array([[1], [2, 3]])
    b = ak.Array([[1, 2], [3]])
    assert not ak.array_equal(a, b)

    # different outer index, same inner index
    a = ak.Array([[[], [1]], [[], [0]]])
    b = ak.Array([[[], [1], []], [[0]]])
    assert not ak.array_equal(a, b)

    # same outer index, different inner index
    a = ak.Array([[[0, 1], [1]], [[0], []]])
    b = ak.Array([[[0], [1, 1]], [[], [0]]])
    assert not ak.array_equal(a, b)

    # nested
    a = ak.Array([[[[]], [[0, 1], [1]]], [[[0]], [[]]]])
    b = ak.Array([[[[0]], [[1], [1]]], [[[]], [[0]]]])
    assert not ak.array_equal(a, b)


def test_array_equal_mixed_content_type():
    a1 = ak.Array([[1, 2, 3], [4, 5, 6], [7, 8]])
    a1r = ak.to_regular(a1[:2])
    assert not ak.array_equal(a1[:2], a1r)
    assert ak.array_equal(a1[:2], a1r, check_regular=False)
    assert not ak.array_equal(a1, a1r, check_regular=False)

    assert ak.array_equal(
        a1, a1.layout
    )  # high-level automatically converted to layout in pre-check

    a2_np = ak.contents.NumpyArray(np.arange(3))
    a2_ia = ak.contents.IndexedArray(
        Index64(np.array([0, 1, 2])), NumpyArray(np.arange(3))
    )
    assert ak.array_equal(a2_np, a2_ia, same_content_types=False)


def test_array_equal_opion_types():
    a1 = ak.Array([1, 2, None, 4])
    a2 = ak.concatenate([ak.Array([1, 2]), ak.Array([None, 4])])
    assert ak.array_equal(a1, a2)

    a3 = a1.layout.to_ByteMaskedArray(valid_when=True)
    assert not ak.array_equal(a1, a3, same_content_types=True)
    assert ak.array_equal(a1, a3, same_content_types=False)
    assert not ak.array_equal(
        a1, ak.Array([1, 2, 3, 4]), same_content_types=False, dtype_exact=False
    )


def test_array_equal_nan():
    a = ak.Array([1.0, 2.5, np.nan])
    nplike = a.layout.backend.nplike
    assert not nplike.array_equal(a.layout.data, a.layout.data)
    assert nplike.array_equal(a.layout.data, a.layout.data, equal_nan=True)
    assert not ak.array_equal(a, a)
    assert ak.array_equal(a, a, equal_nan=True)


def test_array_equal_with_params():
    a1 = NumpyArray(
        np.array([1, 2, 3], dtype=np.uint32), parameters={"foo": {"bar": "baz"}}
    )
    a2 = NumpyArray(
        np.array([1, 2, 3], dtype=np.uint32), parameters={"foo": {"bar": "Not so fast"}}
    )
    assert not ak.array_equal(a1, a2)
    assert ak.array_equal(a1, a2, check_parameters=False)


def test_array_equal_numpy_override():
    assert np.array_equal(
        ak.Array([[1, 2], [], [3, 4, 5]]),
        ak.Array([[1, 2], [], [3, 4, 5]]),
    )
