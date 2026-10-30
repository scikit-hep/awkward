# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def mixed_union():
    # 2 * union[int64, var * int64]: branches have different depths
    return ak.Array([1, [2, 3]]).layout


def test_bare_mixed_union_is_minus_one():
    assert mixed_union().purelist_depth == -1
    assert mixed_union().form.purelist_depth == -1


@pytest.mark.parametrize(
    "wrap",
    [
        lambda x: ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 1, 2], dtype=np.int64)), x
        ),
        lambda x: ak.contents.ListArray(
            ak.index.Index64(np.array([0, 1], dtype=np.int64)),
            ak.index.Index64(np.array([1, 2], dtype=np.int64)),
            x,
        ),
        lambda x: ak.contents.RegularArray(x, 1),
    ],
    ids=["ListOffsetArray", "ListArray", "RegularArray"],
)
def test_list_of_mixed_union_propagates_minus_one(wrap):
    layout = wrap(mixed_union())
    assert layout.purelist_depth == -1
    assert layout.form.purelist_depth == -1
    # nesting twice must not climb back to a positive depth
    assert wrap(layout).purelist_depth == -1
    assert wrap(layout).form.purelist_depth == -1


def test_ndim_of_list_of_mixed_union():
    # previously -1 + 1 == 0, as if the array were a scalar
    assert ak.Array([[[1]], [2]]).ndim == -1
    assert ak.Array([1, [2]]).ndim == -1


def test_strings_are_still_depth_one():
    assert ak.Array(["a", "bc"]).layout.purelist_depth == 1
    assert ak.Array([["a"], ["bc", "d"]]).layout.purelist_depth == 2


def test_broadcast_into_nested_mixed_union():
    # 2 * union[int64, var * var * union[int64, var * int64]]
    b = ak.Array([1, [[2, [3]]]])
    assert b.layout.purelist_depth == -1

    left, right = ak.broadcast_arrays(b, ak.Array([10, 20]))
    assert left.to_list() == [1, [[2, [3]]]]
    assert right.to_list() == [10, [[20, [20]]]]


def test_broadcast_into_mixed_union_unchanged():
    a = ak.Array([1, [2, 3]])
    left, right = ak.broadcast_arrays(a, ak.Array([10, 20]))
    assert left.to_list() == [1, [2, 3]]
    assert right.to_list() == [10, [20, 20]]
