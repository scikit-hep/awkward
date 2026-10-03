# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")

to_list = ak.operations.to_list


def test_union_option_child_longer_than_used_length_4228():
    c, i = ak.contents, ak.index
    layout = c.UnionArray(
        i.Index8(np.array([0, 1], dtype=np.int8)),
        i.Index64(np.array([0, 0], dtype=np.int64)),
        [
            c.ByteMaskedArray(
                i.Index8(np.array([1], dtype=np.int8)),
                c.NumpyArray(np.array([9.9])),
                valid_when=True,
            ),
            c.IndexedOptionArray(
                i.Index64(np.array([0, -1], dtype=np.int64)),
                ak.to_layout([[1.0, 2.0]]),
            ),
        ],
    )
    akarray = ak.Array(layout)
    assert ak.to_arrow(akarray).to_pylist() == to_list(akarray)


def test_union_skipped_index_under_option_4228():
    c, i = ak.contents, ak.index
    union = c.UnionArray(
        i.Index8(np.array([1, 0], dtype=np.int8)),
        i.Index64(np.array([1, 0], dtype=np.int64)),
        [c.NumpyArray(np.array([9.9])), ak.to_layout(["a", "b"])],
    )
    akarray = ak.Array(
        c.ByteMaskedArray(
            i.Index8(np.array([1, 0], dtype=np.int8)),
            c.RecordArray([union], ["x"]),
            valid_when=True,
        )
    )
    assert ak.to_arrow(akarray).to_pylist() == to_list(akarray)


@pytest.mark.parametrize(
    "akarray",
    [
        ak.Array([1, "x", None])[:0],
        ak.Array(["a", None, 1])[:1],
        ak.Array([[1, None, "x"], []])[1:],
        ak.Array([{"u": 1}, {"u": "x"}, {"u": None}])[:0],
    ],
)
def test_sliced_union_with_option_children_4228(akarray):
    assert ak.to_arrow(akarray).to_pylist() == to_list(akarray)
