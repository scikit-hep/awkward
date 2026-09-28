# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize(
    ("dtype", "cls"),
    [
        (np.int8, ak.index.Index8),
        (np.uint8, ak.index.IndexU8),
        (np.int32, ak.index.Index32),
        (np.uint32, ak.index.IndexU32),
        (np.int64, ak.index.Index64),
    ],
)
def test_dtype_dispatch(dtype, cls):
    index = ak.index.Index(np.arange(5, dtype=dtype))
    assert type(index) is cls
    assert index.dtype == np.dtype(dtype)


def test_longlong_is_viewed_as_int64():
    index = ak.index.Index(np.arange(5, dtype=np.longlong))
    assert type(index) is ak.index.Index64
    assert index.dtype == np.dtype(np.int64)


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.uint64])
def test_unsupported_dtype_raises(dtype):
    with pytest.raises(TypeError, match="Index data must be int8, uint8, int32"):
        ak.index.Index(np.arange(5, dtype=dtype))


def test_multidimensional_raises():
    with pytest.raises(TypeError, match="Index data must be one-dimensional"):
        ak.index.Index(np.arange(6, dtype=np.int64).reshape(2, 3))


def test_whole_slice_returns_self():
    index = ak.index.Index(np.arange(10, dtype=np.int64))
    # only the fully-normalised whole-array slice takes the shortcut
    assert index[0 : len(index) : 1] is index
    # while an equivalent, unnormalised slice still copies out a new Index
    assert index[:] is not index
    assert index[:].data.tolist() == list(range(10))


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        (slice(2, 5), [2, 3, 4]),
        (slice(None, None, 2), [0, 2, 4, 6, 8]),
        (slice(None, None, -1), list(range(9, -1, -1))),
        (slice(0, 100, 1), list(range(10))),
        (slice(0, 10, 2), [0, 2, 4, 6, 8]),
    ],
)
def test_partial_slices_make_a_new_index(where, expected):
    index = ak.index.Index(np.arange(10, dtype=np.int64))
    out = index[where]
    assert out is not index
    assert type(out) is ak.index.Index64
    assert out.data.tolist() == expected


def test_typetracer_index_has_no_shortcut():
    layout = ak.Array([[1, 2, 3], [], [4, 5]]).layout.to_typetracer(forget_length=True)
    offsets = layout.offsets
    assert not offsets.nplike.known_data
    out = offsets[0 : offsets.length : 1]
    assert out is not offsets
    assert type(out) is ak.index.Index64
    assert out.length is ak.typetracer.unknown_length
