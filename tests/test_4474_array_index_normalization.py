# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._backends.numpy import NumpyBackend


def list_call(dtype, advanced, starts, stops, indices, mapping=None, lencontent=11):
    starts = np.asarray(starts, dtype=dtype)
    stops = np.asarray(stops, dtype=dtype)
    indices = np.asarray(indices, dtype=np.int64)
    count = len(starts) if advanced else len(starts) * len(indices)
    carry = np.full(count, -99, dtype=np.int64)
    nextadvanced = np.full(count, -99, dtype=np.int64)
    name = "awkward_ListArray_getitem_next_array" + ("_advanced" if advanced else "")
    key = (name, np.int64, np.int64, dtype, dtype, np.int64)
    kernel = NumpyBackend.instance()[key + ((np.int64,) if advanced else ())]
    args = [carry, nextadvanced, starts, stops, indices]
    if advanced:
        # The advanced mapping buffer has the same index dtype as fromarray.
        args += [np.asarray(mapping, dtype=np.int64), len(starts), lencontent]
    else:
        args += [len(starts), len(indices), lencontent]
    return kernel(*args), carry, nextadvanced


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("advanced", [False, True])
def test_valid_array_indices(dtype, advanced):
    error, carry, mapping = list_call(
        dtype, advanced, [5, 8], [8, 11], [-3, -1, 0, 2], [3, 0]
    )
    assert error.str is None
    np.testing.assert_array_equal(
        carry, [7, 8] if advanced else [5, 7, 5, 7, 8, 10, 8, 10]
    )
    np.testing.assert_array_equal(mapping, [0, 1] if advanced else [0, 1, 2, 3] * 2)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("advanced", [False, True])
@pytest.mark.parametrize("at", [-4, 3, -(2**63), 2**63 - 1])
def test_list_error_context(dtype, advanced, at):
    # Row zero accepts the first index; row one reports the original bad index.
    indices = [0, at] if advanced else [at]
    error, carry, mapping = list_call(dtype, advanced, [0, 5], [3, 8], indices, [0, 1])
    assert error.str == b"index out of range"
    assert error.id == (1 if advanced else 0)
    assert error.attempt == at
    np.testing.assert_array_equal(carry, [0, -99] if advanced else [-99, -99])
    np.testing.assert_array_equal(mapping, [0, -99] if advanced else [-99, -99])


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("advanced", [False, True])
@pytest.mark.parametrize(
    "starts,stops,message",
    [([5], [4], b"stops[i] < starts[i]"), ([5], [12], b"stops[i] > len(content)")],
)
def test_structural_error_precedes_index_error(dtype, advanced, starts, stops, message):
    error, carry, mapping = list_call(dtype, advanced, starts, stops, [2**63 - 1], [0])
    assert error.str == message
    assert error.id == 0
    assert error.attempt == 2**63 - 1
    assert carry[0] == mapping[0] == -99


@pytest.mark.parametrize("at", [-3, 2, -(2**63), 2**63 - 1])
def test_regular_partial_write(at):
    kernel = NumpyBackend.instance()[
        "awkward_RegularArray_getitem_next_array_regularize", np.int64, np.int64
    ]
    source = np.asarray([-1, at, 0], dtype=np.int64)
    target = np.full(3, -99, dtype=np.int64)
    error = kernel(target, source, 3, 2)
    assert error.str == b"index out of range"
    assert error.id == 2**63 - 1
    assert error.attempt == at
    np.testing.assert_array_equal(target, [1, at + 2 if at < 0 else at, -99])


@pytest.mark.parametrize("regular", [False, True])
def test_public_array_slicing(regular):
    source = ak.Array([[0, 1, 2], [3, 4, 5]])
    if regular:
        source = ak.to_regular(source)
    assert source[:, [-3, -1, 0, 2]].to_list() == [[0, 2, 0, 2], [3, 5, 3, 5]]
    assert source[[0, 1], [-1, -3]].to_list() == [2, 3]
    assert source[:, []].to_list() == [[], []]
    with pytest.raises(IndexError, match="index out of range"):
        source[:, [3]]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("advanced", [False, True])
def test_empty_outer_array(dtype, advanced):
    error, carry, mapping = list_call(dtype, advanced, [], [], [2**63 - 1], [])
    assert error.str is None
    assert len(carry) == len(mapping) == 0


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_empty_index_preserves_structural_validation(dtype):
    # Empty lists may point beyond content; no index means no content access.
    error, carry, mapping = list_call(dtype, False, [99], [99], [])
    assert error.str is None
    assert len(carry) == len(mapping) == 0
    error, _, _ = list_call(dtype, False, [5], [4], [])
    assert error.str == b"stops[i] < starts[i]"


def test_regular_valid_and_empty_indices():
    kernel = NumpyBackend.instance()[
        "awkward_RegularArray_getitem_next_array_regularize", np.int64, np.int64
    ]
    source = np.asarray([-2, -1, 0, 1], dtype=np.int64)
    target = np.full(4, -99, dtype=np.int64)
    assert kernel(target, source, 4, 2).str is None
    np.testing.assert_array_equal(target, [0, 1, 0, 1])
    assert kernel(target, source, 0, 0).str is None
    np.testing.assert_array_equal(target, [0, 1, 0, 1])
