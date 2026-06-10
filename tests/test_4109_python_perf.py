# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def _make_union(*arrays):
    out = ak.concatenate([ak.Array(a) for a in arrays])
    assert out.layout.is_union
    return out


# ---------------------------------------------------------------------------
# Change 1: vectorized UnionArray._remove_structure
#
# These assert exact-order equivalence with the historical element-by-element
# behaviour. The reference values were captured from the original implementation.
# ---------------------------------------------------------------------------


def test_union_flatten_numbers_and_lists_order():
    union = _make_union([1, 2, 3], [[10, 20], [30]], [4, 5])
    assert to_list(ak.flatten(union, axis=None)) == [1, 2, 3, 10, 20, 30, 4, 5]


def test_union_flatten_with_none_drops_nones_in_order():
    union = _make_union([1, None, 3], [[10, 20], [30]], [4, 5])
    assert to_list(ak.flatten(union, axis=None)) == [1, 3, 10, 20, 30, 4, 5]


def test_union_flatten_nested_none_everywhere():
    union = _make_union([1, None, 3], [[10, None], [30]], [None, 5])
    assert to_list(ak.flatten(union, axis=None)) == [1, 3, 10, 30, 5]


def test_union_flatten_deeply_nested_lists():
    union = _make_union([[[1], [2, 3]]], [4, 5], [[6, 7, 8]])
    assert to_list(ak.flatten(union, axis=None)) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_union_flatten_records_fallback_preserves_order():
    # Records under flatten_records produce multiple parts per element, so the
    # implementation falls back to the per-element loop. Order must be preserved.
    union = _make_union(
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        [100, 200],
        [{"x": 5, "y": 6}],
    )
    assert to_list(ak.flatten(union, axis=None)) == [1, 2, 3, 4, 100, 200, 5, 6]


def test_union_remove_structure_matches_reference_many():
    rng = np.random.default_rng(12345)
    for _ in range(50):
        n_chunks = rng.integers(2, 5)
        chunks = []
        for _ in range(n_chunks):
            kind = rng.integers(0, 3)
            size = int(rng.integers(0, 6))
            if kind == 0:
                vals = rng.integers(0, 100, size=size).tolist()
                # sprinkle Nones
                vals = [None if rng.random() < 0.2 else v for v in vals]
                chunks.append(ak.Array(vals))
            elif kind == 1:
                chunks.append(
                    ak.Array(
                        [rng.integers(0, 100, int(rng.integers(0, 4))).tolist()] * size
                    )
                )
            else:
                chunks.append(ak.Array([[[1, 2], [3]]] * size))
        try:
            union = ak.concatenate(chunks)
        except Exception:
            continue
        if not union.layout.is_union:
            continue
        got = to_list(ak.flatten(union, axis=None))

        # Reference: explicit per-element flattening of the same union.
        layout = union.layout
        tags = np.asarray(layout.tags.data)
        index = np.asarray(layout.index.data)
        ref = []
        for i in range(len(tags)):
            sub = layout.contents[tags[i]][index[i] : index[i] + 1]
            parts = ak._do.remove_structure(sub, function_name="ak.flatten")
            for p in parts:
                ref.extend(to_list(ak.Array(p)))
        assert got == ref


def test_union_flatten_reduction_over_heterogeneous():
    union = _make_union([1, 2, 3], [[10, 20], [30]], [4, 5])
    # flatten(axis=None) then reduce
    assert ak.sum(ak.flatten(union, axis=None)) == 75
    # direct axis=None reduction over a union that flattens to a single array
    assert ak.sum(union, axis=None) == 75
    assert ak.max(union, axis=None) == 30


def test_union_flatten_typetracer():
    union = _make_union([1, 2, 3], [[10, 20], [30]], [4, 5])
    tt = ak.Array(union.layout.to_typetracer(forget_length=True))
    result = ak.flatten(tt, axis=None)
    # type is preserved, lengths are unknown but the op must not raise
    assert str(result.type.content) == "int64"


def test_union_flatten_empty_tag():
    # A union where one content contributes no elements.
    union = _make_union([1, 2, 3], [[10, 20], [30]])
    union = union[union.layout.tags.data == 0]
    if union.layout.is_union:
        assert to_list(ak.flatten(union, axis=None)) == [1, 2, 3]
    else:
        # simplification may collapse to a single content; still must flatten
        assert to_list(ak.flatten(union, axis=None)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Change 2: cached unique equal_nan probe
# ---------------------------------------------------------------------------


def test_unique_equal_nan_probe_is_cached():
    from awkward._nplikes.array_module import _nplike_unique_has_equal_nan

    _nplike_unique_has_equal_nan.cache_clear()
    info0 = _nplike_unique_has_equal_nan.cache_info()
    assert info0.currsize == 0

    a = ak.Array([3, 1, 2, 1, 3, 2])
    ak.sort(a)  # uses sort, not unique; just keep array alive
    # Trigger unique_values several times via the nplike directly
    nplike = a.layout.backend.nplike
    for _ in range(5):
        nplike.unique_values(a.layout.data)
    info = _nplike_unique_has_equal_nan.cache_info()
    # exactly one miss (first probe), the rest are cache hits
    assert info.misses == 1
    assert info.hits >= 4


def test_unique_values_still_correct():
    nplike = ak.Array([1]).layout.backend.nplike
    data = np.array([3, 1, 2, 1, 3, 2, 5], dtype=np.int64)
    result = np.asarray(nplike.unique_values(data))
    assert to_list(result) == [1, 2, 3, 5]


# ---------------------------------------------------------------------------
# Change 3: subrange_equal without redundant buffer copy
# ---------------------------------------------------------------------------


def test_is_unique_detects_duplicate_subranges():
    # Equal subranges -> not unique (exercises subrange_equal kernel)
    assert ak._do.is_unique(ak.Array([[1, 2, 3], [1, 2, 3], [4, 5]]).layout) is False
    # All distinct -> unique
    assert ak._do.is_unique(ak.Array([[1, 2, 3], [4, 5, 6]]).layout) is True


def test_is_unique_with_options():
    assert ak._do.is_unique(ak.Array([[1, 2], [1, 2], None]).layout) is False
    assert ak._do.is_unique(ak.Array([[1, 2], [3, 4], None]).layout) is True


def test_is_unique_bool_subranges():
    # Exercises the awkward_NumpyArray_subrange_equal_bool kernel: identical
    # subranges must be detected as non-unique.
    assert (
        ak._do.is_unique(ak.Array([[True, False], [True, False], [True]]).layout)
        is False
    )


def test_is_unique_non_contiguous_buffer():
    # A non-contiguous NumpyArray must still be handled by subrange_equal.
    base = np.arange(20, dtype=np.int64).reshape(10, 2)
    col = ak.Array(base[:, 0])  # non-contiguous view
    grouped = ak.unflatten(col, [3, 3, 4])
    assert ak._do.is_unique(grouped.layout) is True
