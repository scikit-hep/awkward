# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_indexed():
    layout = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np.array([0, 3], dtype=np.int64)),
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index(np.array([0, 1, 2], dtype=np.int64)),
            ak._v2.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64)),
        ),
    )

    assert ak._v2.operations.describe.is_simplified(layout)


def test_indexedoption():
    layout = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np.array([0, 3], dtype=np.int64)),
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index(np.array([0, 1, -1], dtype=np.int64)),
            ak._v2.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64)),
        ),
    )

    assert ak._v2.operations.describe.is_simplified(layout)


def test_indexed_indexedoption():
    layout = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np.array([0, 3], dtype=np.int64)),
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index(np.array([0, 1, 2], dtype=np.int64)),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index(np.array([0, 1, -1], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64)),
            ),
        ),
    )

    assert not ak._v2.operations.describe.is_simplified(layout)


def test_indexedoption_indexed():
    layout = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np.array([0, 3], dtype=np.int64)),
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index(np.array([-1, 1, 2], dtype=np.int64)),
            ak._v2.contents.IndexedArray(
                ak._v2.index.Index(np.array([0, 1, 2], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64)),
            ),
        ),
    )

    assert not ak._v2.operations.describe.is_simplified(layout)


def test_indexedoption_indexedoption():
    layout = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np.array([0, 3], dtype=np.int64)),
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index(np.array([-1, 1, 2], dtype=np.int64)),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index(np.array([0, 1, -1], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64)),
            ),
        ),
    )

    assert not ak._v2.operations.describe.is_simplified(layout)


def test_indexed_indexed():
    layout = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np.array([0, 3], dtype=np.int64)),
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index(np.array([0, 1, 2], dtype=np.int64)),
            ak._v2.contents.IndexedArray(
                ak._v2.index.Index(np.array([0, 2, 1], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([4, 5, 6], dtype=np.int64)),
            ),
        ),
    )

    assert not ak._v2.operations.describe.is_simplified(layout)


def test_union_mergeable():
    layout = ak._v2.contents.UnionArray(
        ak._v2.index.Index(np.array([0, 0, 0, 0], dtype=np.int8)),
        ak._v2.index.Index(np.array([0, 1, 0, 1], dtype=np.int64)),
        [
            ak._v2.contents.NumpyArray(np.array([0, 1, 2, 3], dtype=np.int64)),
            ak._v2.contents.NumpyArray(np.array([9, 8, 7, 6], dtype=np.int64)),
        ],
    )

    assert not ak._v2.operations.describe.is_simplified(layout)


def test_union_unmergeable():
    layout = ak._v2.contents.UnionArray(
        ak._v2.index.Index(np.array([0, 0, 0, 0], dtype=np.int8)),
        ak._v2.index.Index(np.array([0, 1, 0, 1], dtype=np.int64)),
        [
            ak._v2.contents.NumpyArray(np.array([0, 1, 2, 3], dtype=np.int64)),
            ak._v2.contents.RegularArray(
                ak._v2.contents.NumpyArray(np.array([9, 8, 7, 6], dtype=np.int64)), 1
            ),
        ],
    )
    assert ak._v2.operations.describe.is_simplified(layout)
