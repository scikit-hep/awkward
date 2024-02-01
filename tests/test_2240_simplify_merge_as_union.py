# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_many():
    result = ak.concatenate(
        [ak.Array(x) for x in [[{"a": 3}], [{"c": 3}], [{"d": 3}], [{"e": 3}]]]
    )
    assert result.tolist() == [{"a": 3}, {"c": 3}, {"d": 3}, {"e": 3}]


def test_validity_error_simple():
    layout = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 2, 3, 0, 1, 2, 3, 4], dtype=np.int64)),
        [ak.to_layout([1, 2, 3, 4]), ak.to_layout([5, 6, 7, 8, 9])],
    )
    assert layout.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert "content(1) is mergeable with content(0)" in ak.validity_error(layout)


def test_validity_error_complex():
    layout = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int8)),
        ak.index.Index64(np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=np.int64)),
        [
            ak.to_layout([1, 2, 3, 4]),
            ak.to_layout(["a", "b", "c", "d"]),
            ak.to_layout([5, 6, 7]),
        ],
    )
    assert layout.to_list() == [1, 2, 3, 4, "a", "b", "c", "d", 5, 6, 7]
    assert "content(2) is mergeable with content(0)" in ak.validity_error(layout)
