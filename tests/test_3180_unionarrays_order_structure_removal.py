# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
# ruff: noqa: E402

from __future__ import annotations

import awkward as ak


def test():
    x = ak.Array([None, [False], [True]])
    y = ak.ravel(ak.fill_none(x, False, axis=0))
    assert y.tolist() == [False, False, True]


def test_ListOffsetArray():
    a = ak.Array([[1, 2, 3], 4, [5, 6], 7, 8])
    b = ak.ravel(a)
    assert b.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
