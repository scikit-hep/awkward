# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def make_layout(kind, size=2):
    content = ak.contents.NumpyArray(np.arange(2 * size, dtype=np.int64))
    if kind == "regular":
        return ak.contents.RegularArray(content, size, zeros_length=2)
    index = {
        "i32": ak.index.Index32,
        "u32": ak.index.IndexU32,
        "i64": ak.index.Index64,
    }[kind]
    return ak.contents.ListArray(index([0, size]), index([size, 2 * size]), content)


@pytest.mark.parametrize("kind", ["regular", "i32", "u32", "i64"])
@pytest.mark.parametrize("at", [-2, -1, 0, 1])
def test_valid_scalar_index(kind, at):
    array = ak.Array(make_layout(kind))
    assert array[:, at].to_list() == [[0, 1][at], [2, 3][at]]
    assert array.layout.to_typetracer()[:, at].form == array[:, at].layout.form


@pytest.mark.parametrize("kind", ["regular", "i32", "u32", "i64"])
@pytest.mark.parametrize("at", [-3, 2, -(2**63), 2**63 - 1])
def test_scalar_index_out_of_bounds(kind, at):
    array = ak.Array(make_layout(kind))
    with pytest.raises(IndexError, match="index out of range"):
        array[:, at]


@pytest.mark.parametrize("kind", ["regular", "i32", "u32", "i64"])
@pytest.mark.parametrize("at", [-1, 0])
def test_empty_list_scalar_index(kind, at):
    with pytest.raises(IndexError, match="index out of range"):
        ak.Array(make_layout(kind, size=0))[:, at]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_list_kernel_error_context_and_partial_output(dtype):
    from awkward._backends.numpy import NumpyBackend

    kernel = NumpyBackend.instance()[
        "awkward_ListArray_getitem_next_at", np.int64, dtype, dtype
    ]
    starts = np.asarray([3, 7], dtype=dtype)
    stops = np.asarray([5, 7], dtype=dtype)
    carry = np.full(2, -99, dtype=np.int64)
    error = kernel(carry, starts, stops, 2, -1)
    assert error.str == b"index out of range"
    assert error.id == 1
    assert error.attempt == -1
    assert b"awkward_ListArray_getitem_next_at.cpp" in error.filename
    np.testing.assert_array_equal(carry, [4, -99])
    # With no rows, ListArray does not validate the scalar index.
    assert kernel(carry, starts, stops, 0, 2**63 - 1).str is None


def test_regular_kernel_empty_outer_error_context():
    from awkward._backends.numpy import NumpyBackend

    kernel = NumpyBackend.instance()["awkward_RegularArray_getitem_next_at", np.int64]
    carry = np.full(1, -99, dtype=np.int64)
    assert kernel(carry, -2, 0, 2).str is None
    # RegularArray validates the index even when there are no outer rows.
    error = kernel(carry, 2, 0, 2)
    assert error.str == b"index out of range"
    assert error.id == 2**63 - 1
    assert error.attempt == 2
    assert b"awkward_RegularArray_getitem_next_at.cpp" in error.filename
    assert carry[0] == -99
