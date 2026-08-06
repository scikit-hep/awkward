# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import types

import numpy as np
import pytest

import awkward as ak


def test_softmax_default_axis():
    # Previously ak.softmax with the default axis raised TypeError because the
    # default was axis=None, which maybe_posaxis cannot handle.
    array = ak.Array([[1.0, 2.0, 3.0], [4.0], [], [5.0, 6.0]])
    result = ak.softmax(array)

    # Compare against a numpy/scipy-style softmax computed per inner list.
    expected = []
    for row in array.to_list():
        if len(row) == 0:
            expected.append([])
            continue
        arr = np.array(row)
        e = np.exp(arr)
        expected.append((e / e.sum()).tolist())

    result_list = result.to_list()
    assert len(result_list) == len(expected)
    for got, exp in zip(result_list, expected, strict=True):
        assert np.allclose(got, exp)

    # axis=-1 is the same as the default
    assert ak.softmax(array, axis=-1).to_list() == result_list


def test_softmax_axis_none_raises():
    array = ak.Array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        ak.softmax(array, axis=None)


def test_enforce_type_regular_size_equal():
    # 2 * int32 -> 2 * int64 should be enforceable (recurse into content)
    layout = ak.to_regular(ak.Array(np.arange(6, dtype=np.int32).reshape(3, 2))).layout
    from awkward.operations.ak_enforce_type import _type_is_enforceable

    target = ak.types.from_datashape("2 * int64", highlevel=False)
    result = _type_is_enforceable(layout, target)
    assert result.is_enforceable

    out = ak.enforce_type(ak.Array(layout), "2 * int64")
    assert str(out.type) == "3 * 2 * int64"


def test_enforce_type_regular_size_mismatch():
    layout = ak.to_regular(ak.Array(np.arange(6, dtype=np.int32).reshape(3, 2))).layout
    from awkward.operations.ak_enforce_type import _type_is_enforceable

    target = ak.types.from_datashape("3 * int64", highlevel=False)
    result = _type_is_enforceable(layout, target)
    assert not result.is_enforceable

    with pytest.raises(ValueError):
        ak.enforce_type(ak.Array(layout), "3 * int64")


def test_merge_union_of_records_categorical_indexed():
    index = ak.index.Index64(np.array([1, 0, 1], dtype=np.int64))
    inner_records = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.array([10.0, 20.0]))], ["c"]
    )
    indexed = ak.contents.IndexedArray(
        index, inner_records, parameters={"__array__": "categorical"}
    )

    other = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64))], ["a"]
    )

    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8))
    union_index = ak.index.Index64(np.array([0, 0, 1, 1, 2], dtype=np.int64))
    union = ak.contents.UnionArray(tags, union_index, [other, indexed])

    arr = ak.Array(union)
    # Previously raised IndexError due to self-indexing the inner index.
    out = ak.merge_union_of_records(arr)
    out_list = out.to_list()

    # tag 0 -> other (field a); tag 1 -> categorical indexed records (field c).
    # The inner categorical index [1, 0, 1] selects content [20.0, 10.0, 20.0],
    # consumed in order by the two tag-1 slots (union_index 0 and 1).
    assert out_list == [
        {"a": 1, "c": None},
        {"a": None, "c": 20.0},
        {"a": 2, "c": None},
        {"a": None, "c": 10.0},
        {"a": None, "c": None},
    ]
    assert str(out.type) == "5 * {a: ?int64, c: ?float64}"


def test_argcartesian_list_vs_dict_parity_axis1():
    x = ak.Array([[1, 2, 3], [4]])
    y = ak.Array([[10, 20], [30]])

    as_dict = ak.argcartesian({"x": x, "y": y}, axis=1)
    as_list = ak.argcartesian([x, y], axis=1)

    # Field names differ ("x"/"y" vs "0"/"1") but the index values must match.
    dict_vals = [[(d["x"], d["y"]) for d in row] for row in as_dict.to_list()]
    list_vals = [[tuple(t) for t in row] for row in as_list.to_list()]
    assert dict_vals == list_vals


def test_from_raggedtensor_recursive_deep():
    # Drive _recursive_call directly (no TensorFlow needed) for a structure with
    # three ragged dimensions. Before the fix the recursive else-branch reused
    # the same `count`, causing a RecursionError for >= 3 ragged dimensions.
    from awkward.operations.ak_from_raggedtensor import _recursive_call

    content = ak.contents.NumpyArray(np.arange(8, dtype=np.float64))
    offsets = [
        ak.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak.index.Index64(np.array([0, 2, 4], dtype=np.int64)),
        ak.index.Index64(np.array([0, 2, 4, 6, 8], dtype=np.int64)),
    ]
    result = ak.Array(_recursive_call(content, offsets, 0))
    assert result.to_list() == [[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]]


def _make_dispatch_sentinel():
    """Array-like that intercepts __awkward_function__ dispatch and records arguments."""
    captured = []

    def __awkward_function__(func, array_likes, args, kwargs):
        captured[:] = list(array_likes)
        return "intercepted"

    sentinel = types.SimpleNamespace(
        captured=captured,
        __awkward_function__=__awkward_function__,
    )
    return sentinel


def test_unflatten_dispatch_sees_counts():
    array = ak.Array([1, 2, 3, 4])
    sentinel = _make_dispatch_sentinel()
    # counts is array-like and must be offered to dispatch.
    result = ak.unflatten(array, sentinel)
    assert result == "intercepted"
    assert any(x is sentinel for x in sentinel.captured)


def test_nan_to_num_dispatch_sees_extra_args():
    array = ak.Array([1.0, 2.0])
    sentinel = _make_dispatch_sentinel()
    # Pass the dispatcher as posinf; dispatch must offer it.
    result = ak.nan_to_num(array, posinf=sentinel)
    assert result == "intercepted"
    assert any(x is sentinel for x in sentinel.captured)
