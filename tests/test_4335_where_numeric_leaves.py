# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.placeholder import PlaceholderArray

condition = ak.Array([[True, False, True], [], [False, True]])
x = ak.Array([[1, 2, 3], [], [4, 5]])
y = ak.Array([[10.5, 20.5, 30.5], [], [40.5, 50.5]])


def flat(array):
    return np.asarray(ak.flatten(array, axis=None))


def test_numeric_leaves_match_numpy():
    result = ak.where(condition, x, y)
    assert result.ndim == 2
    assert result.to_list() == [[1.0, 20.5, 3.0], [], [40.5, 5.0]]
    expected = np.where(flat(condition), flat(x), flat(y))
    assert flat(result).tolist() == expected.tolist()
    assert flat(result).dtype == expected.dtype


@pytest.mark.parametrize("scalar", [0, -1.5, np.float32(2)])
def test_broadcast_scalar_branch(scalar):
    for result, expected in [
        (ak.where(condition, x, scalar), np.where(flat(condition), flat(x), scalar)),
        (ak.where(condition, scalar, x), np.where(flat(condition), scalar, flat(x))),
    ]:
        assert flat(result).tolist() == expected.tolist()
        assert flat(result).dtype == expected.dtype


@pytest.mark.parametrize(
    ("x_dtype", "y_dtype"),
    [("int8", "int64"), ("int32", "float32"), ("uint8", "float64"), ("bool", "int16")],
)
def test_dtype_promotion_matches_numpy(x_dtype, y_dtype):
    left = ak.values_astype(x, x_dtype)
    right = ak.values_astype(x, y_dtype)
    result = ak.where(condition, left, right)
    expected = np.where(flat(condition), flat(left), flat(right))
    assert flat(result).dtype == expected.dtype
    assert flat(result).tolist() == expected.tolist()


def test_mergebool_false_still_builds_a_union():
    booleans = ak.values_astype(condition, "bool")
    assert str(ak.where(condition, booleans, x).type) == "3 * var * int64"
    assert (
        str(ak.where(condition, booleans, x, mergebool=False).type)
        == "3 * var * union[bool, int64]"
    )


def test_option_leaf_falls_back():
    # an IndexedOptionArray leaf is a Content that is not a NumpyArray
    optional = ak.Array([[1, None, 3], [], [4, 5]])
    result = ak.where(condition, optional, y)
    assert result.to_list() == [[1, 20.5, 3], [], [40.5, 5]]
    assert str(result.type) == "3 * var * ?float64"


@pytest.mark.parametrize(
    ("other", "type_string"),
    [
        (ak.Array([["a", "b", "c"], [], ["d", "e"]]), "3 * var * union[string, int64]"),
        (
            ak.Array([[[1], [2], [3]], [], [[4], [5]]]),
            "3 * var * union[var * int64, int64]",
        ),
        (
            ak.Array([[{"n": 1}] * 3, [], [{"n": 1}] * 2]),
            "3 * var * union[{n: int64}, int64]",
        ),
    ],
)
def test_non_numpyarray_leaf_falls_back(other, type_string):
    # covers the `isinstance(obj, Content)` bail-out of the fast path
    result = ak.where(condition, other, x)
    assert str(result.type) == type_string


def test_scalar_against_non_numeric_leaf():
    strings = ak.Array([["a", "b", "c"], [], ["d", "e"]])
    assert ak.where(condition, 5, strings).to_list() == [[5, "b", 5], [], ["d", 5]]
    assert ak.where(condition, strings, 5).to_list() == [["a", 5, "c"], [], [5, "e"]]


def test_foreign_buffer_falls_back():
    # a PlaceholderArray is not the backend's own array type, so the fast path
    # must bail out rather than hand it to `nplike.where`
    placeholder = ak.Array(
        ak.from_buffers(
            {"class": "NumpyArray", "primitive": "int64", "form_key": "node0"},
            3,
            {"node0-data": PlaceholderArray(Numpy.instance(), (3,), np.int64)},
            highlevel=False,
        )
    )
    strings = ak.Array(["a", "b", "c"])
    result = ak.where(np.array([True, False, True]), placeholder, strings)
    assert str(result.type) == "3 * union[int64, string]"
