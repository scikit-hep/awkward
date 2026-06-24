# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak

to_list = ak.operations.to_list


def _make_union(*arrays):
    out = ak.concatenate([ak.Array(a) for a in arrays])
    assert out.layout.is_union
    return out


def test_union_remove_structure_matches_reference():
    # Vectorized fast path must match the historical per-element flattening
    # exactly, including ordering and per-leaf dropping of Nones.
    union = _make_union([1, None, 3], [[10, None], [30]], [None, 5])

    layout = union.layout
    tags = np.asarray(layout.tags.data)
    index = np.asarray(layout.index.data)
    ref = []
    for i in range(len(tags)):
        sub = layout.contents[tags[i]][index[i] : index[i] + 1]
        for p in ak._do.remove_structure(sub, function_name="ak.flatten"):
            ref.extend(to_list(ak.Array(p)))

    assert to_list(ak.flatten(union, axis=None)) == ref == [1, 3, 10, 30, 5]


def test_union_flatten_records_fallback_preserves_order():
    # Records under flatten_records produce multiple parts per element, so the
    # implementation falls back to the per-element loop. Order must be preserved.
    union = _make_union(
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        [100, 200],
        [{"x": 5, "y": 6}],
    )
    assert to_list(ak.flatten(union, axis=None)) == [1, 2, 3, 4, 100, 200, 5, 6]


def test_unique_values_still_correct():
    nplike = ak.Array([1]).layout.backend.nplike
    data = np.array([3, 1, 2, 1, 3, 2, 5], dtype=np.int64)
    result = np.asarray(nplike.unique_values(data))
    assert to_list(result) == [1, 2, 3, 5]


def test_is_unique_subranges():
    assert ak._do.is_unique(ak.Array([[1, 2, 3], [1, 2, 3], [4, 5]]).layout) is False
    assert ak._do.is_unique(ak.Array([[1, 2, 3], [4, 5, 6]]).layout) is True


def test_is_unique_non_contiguous_buffer():
    # A non-contiguous NumpyArray must still be handled by subrange_equal.
    base = np.arange(20, dtype=np.int64).reshape(10, 2)
    col = ak.Array(base[:, 0])  # non-contiguous view
    grouped = ak.unflatten(col, [3, 3, 4])
    assert ak._do.is_unique(grouped.layout) is True
