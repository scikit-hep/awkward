# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def _list_offset_array(offsets):
    content = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 1, 2, -1, 3], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([10, 20, 30, 40], dtype=np.int64)),
    )
    return ak.Array(ak.contents.ListOffsetArray(offsets, content))


@pytest.mark.parametrize(
    "offsets",
    [
        ak.index.Index32(np.array([0, 2, 2, 6], dtype=np.int32)),
        ak.index.IndexU32(np.array([0, 2, 2, 6], dtype=np.uint32)),
        ak.index.Index64(np.array([0, 2, 2, 6], dtype=np.int64)),
    ],
)
@pytest.mark.parametrize("axis", [None, 1])
def test_drop_none_list_offsets_dtypes(offsets, axis):
    array = _list_offset_array(offsets)
    assert array.to_list() == [[10, None], [], [20, 30, None, 40]]
    assert ak.drop_none(array, axis=axis).to_list() == [[10], [], [20, 30, 40]]


def test_drop_none_outer_and_inner_nones_int32_offsets():
    content = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 1], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([1, 2], dtype=np.int64)),
    )
    layout = ak.contents.ListOffsetArray(
        ak.index.Index32(np.array([0, 2, 3], dtype=np.int32)), content
    )
    array = ak.Array(layout)
    assert ak.drop_none(array, axis=1).to_list() == [[1], [2]]


def test_drop_none_arrow_lists():
    pa = pytest.importorskip("pyarrow")

    # Arrow ``list`` arrays use int32 offsets.
    array = ak.from_arrow(pa.array([[1, 2, None], None, [], [4, 5], [6]]))
    assert ak.drop_none(array, axis=1).to_list() == [[1, 2], None, [], [4, 5], [6]]

    nested = ak.from_arrow(pa.array([[[1, 2], None, [3]], None, [], [[4], [5, None]]]))
    assert ak.drop_none(nested, axis=2).to_list() == [
        [[1, 2], None, [3]],
        None,
        [],
        [[4], [5]],
    ]
