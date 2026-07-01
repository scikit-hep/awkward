# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


# Regression test for the offsets-pipeline migration, which rewrote
# IndexedArray._unique (and _is_unique) from a ~120-line parents-era
# implementation to a "carry-and-delegate" form: project through the index,
# then defer to the content's own _unique/_is_unique. These tests pin the
# invariant that drove the concern -- computing unique on an IndexedArray must
# give exactly the same result as computing it on the materialized
# (projected) array -- across reordering, repetition, both axes, and several
# dtypes, so a future change to the delegation cannot silently regress a
# covered case.


def _indexed(index, content):
    return ak.contents.IndexedArray(
        ak.index.Index64(np.asarray(index, dtype=np.int64)),
        content,
    )


@pytest.mark.parametrize("axis", [None, -1])
def test_indexedarray_unique_matches_projection(axis):
    # Distinct underlying values; the index reorders and repeats them, so the
    # projected array has known duplicates.
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    indexed = _indexed([0, 2, 0, 4, 2], content)

    # Sanity: the projection is what we think it is.
    assert to_list(indexed.project()) == [1.1, 3.3, 1.1, 5.5, 3.3]

    # The delegation must agree with operating on the materialized array...
    assert to_list(ak._do.unique(indexed, axis=axis)) == to_list(
        ak._do.unique(indexed.project(), axis=axis)
    )
    # ...and produce the expected sorted, de-duplicated values. axis=None
    # returns a flat result; axis=-1 keeps the (single) outer dimension.
    expected = [1.1, 3.3, 5.5] if axis is None else [[1.1, 3.3, 5.5]]
    assert to_list(ak._do.unique(indexed, axis=axis)) == expected


def test_indexedarray_is_unique_with_duplicates():
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    indexed = _indexed([0, 2, 0, 4, 2], content)

    assert ak._do.is_unique(indexed) is ak._do.is_unique(indexed.project())
    assert ak._do.is_unique(indexed) is False


def test_indexedarray_is_unique_all_distinct():
    # A pure reordering of distinct values stays unique.
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    indexed = _indexed([4, 3, 2, 1, 0], content)

    assert to_list(indexed.project()) == [5.5, 4.4, 3.3, 2.2, 1.1]
    assert ak._do.is_unique(indexed) is True
    assert to_list(ak._do.unique(indexed)) == [1.1, 2.2, 3.3, 4.4, 5.5]


@pytest.mark.parametrize(
    "dtype", [np.int64, np.int32, np.uint32, np.float32, np.float64]
)
def test_indexedarray_unique_dtypes(dtype):
    content = ak.contents.NumpyArray(np.array([10, 20, 30, 40], dtype=dtype))
    indexed = _indexed([3, 1, 3, 0, 1], content)  # projects to [40, 20, 40, 10, 20]

    assert to_list(ak._do.unique(indexed)) == to_list(ak._do.unique(indexed.project()))
    assert to_list(ak._do.unique(indexed)) == [10, 20, 40]
    assert ak._do.is_unique(indexed) is False


def test_indexedarray_unique_empty():
    # The length-0 fast path must short-circuit cleanly and agree with the
    # projected (also empty) array.
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3]))
    indexed = _indexed([], content)

    assert to_list(ak._do.unique(indexed)) == to_list(ak._do.unique(indexed.project()))
    assert to_list(ak._do.unique(indexed)) == []
    assert ak._do.is_unique(indexed) is True


def test_indexedarray_inside_list_unique_axis_last():
    # Exercise the delegation when the IndexedArray is nested under a list,
    # so _unique is reached with non-trivial offsets (axis=-1 on the leaves).
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    indexed = _indexed([0, 2, 0, 1, 4, 4], content)  # [1.1,3.3,1.1, 2.2,5.5,5.5]
    listed = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 6], dtype=np.int64)),
        indexed,
    )
    projected = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 6], dtype=np.int64)),
        indexed.project(),
    )

    assert to_list(ak._do.unique(listed, axis=-1)) == to_list(
        ak._do.unique(projected, axis=-1)
    )
    assert to_list(ak._do.unique(listed, axis=-1)) == [[1.1, 3.3], [2.2, 5.5]]
