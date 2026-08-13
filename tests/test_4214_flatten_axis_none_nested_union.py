# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_flatten_nested_union():
    array = ak.Array([1, [[2, [3]]]])
    assert to_list(ak.flatten(array, axis=None)) == [1, 2, 3]


def test_flatten_nested_union_trailing_empty_lists():
    # Previously raised IndexError: the merged position marker was longer
    # than the merged values.
    array = ak.Array([1, [[2, [3]]], [], []])
    assert to_list(ak.flatten(array, axis=None)) == [1, 2, 3]


def test_ravel_nested_union():
    # Exercises drop_nones=False.
    array = ak.Array([1, [[2, [3]]]])
    assert to_list(ak.ravel(array)) == [1, 2, 3]


def test_reduce_axis_none_nested_union_raises():
    # Restores the 2.10.0 behavior: reducing over an irreducible union
    # raises. In 2.11.0 this silently returned 3.
    array = ak.Array([1, [[2, [3]]]])
    with pytest.raises(ValueError, match="irreducible unions"):
        ak.sum(array, axis=None)


def _nested_union_branch():
    # union[int64, var * int64]: [100, 200, [1, 2], [3]]
    inner = ak.concatenate([ak.Array([100, 200]), ak.Array([[1, 2], [3]])]).layout
    assert inner.is_union
    return inner


def _outer_union(branch):
    # Elements: branch[0], 999.0, branch[1], 888.0
    return ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 0, 1, 1], dtype=np.int64)),
        [branch, ak.contents.NumpyArray(np.array([999.0, 888.0]))],
    )


def test_explicit_nested_union_silent_truncation():
    # var * var * union[int64, var * int64] has purelist_depth == 1 (the
    # mixed-depth union reports -1), so broadcast_arrays left the position
    # marker unbroadcast; flatten returned [100.0, 1.0, 200.0, 2.0].
    inner = _nested_union_branch()
    lists = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 4], dtype=np.int64)), inner
    )
    nested_lists = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 2], dtype=np.int64)), lists
    )
    array = ak.Array(_outer_union(nested_lists))
    assert to_list(ak.flatten(array, axis=None)) == [
        100.0,
        200.0,
        999.0,
        1.0,
        2.0,
        3.0,
        888.0,
    ]


def test_regular_of_branching_union():
    # Previously raised ValueError from broadcast_arrays: "cannot broadcast
    # UnionArray of length 4 with NumpyArray of length 2".
    inner = _nested_union_branch()
    regular = ak.contents.RegularArray(inner, 2)
    array = ak.Array(_outer_union(regular))
    assert to_list(ak.flatten(array, axis=None)) == [
        100.0,
        200.0,
        999.0,
        1.0,
        2.0,
        3.0,
        888.0,
    ]


def test_matches_per_element_reference():
    # The element-wise fallback must reproduce the historical per-element
    # flattening exactly, interleaving included.
    array = ak.Array([1, [[2, [3]]], 4, [[5, [6]]], 7])

    layout = array.layout
    tags = np.asarray(layout.tags.data)
    index = np.asarray(layout.index.data)
    ref = []
    for i in range(len(tags)):
        sub = layout.contents[tags[i]][index[i] : index[i] + 1]
        for p in ak._do.remove_structure(sub, function_name="ak.flatten"):
            ref.extend(to_list(ak.Array(p)))

    assert to_list(ak.flatten(array, axis=None)) == ref == [1, 2, 3, 4, 5, 6, 7]


def test_flatten_nested_union_typetracer():
    array = ak.Array([1, [[2, [3]]]], backend="typetracer")
    assert ak.flatten(array, axis=None).typestr == "## * int64"
