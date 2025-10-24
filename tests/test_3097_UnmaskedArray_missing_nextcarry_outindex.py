# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
    ListOffsetArray,
    NumpyArray,
    UnmaskedArray,
)
from awkward.index import Index8, Index64, IndexU8


def test_UnmaskedArray():
    layout = ListOffsetArray(
        Index64(np.array([0, 3, 3, 5])),
        UnmaskedArray(NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))),
    )
    out = ak.drop_none(ak.Array(layout), axis=1)
    assert out.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]


def test_ByteMaskedArray():
    layout = ListOffsetArray(
        Index64(np.array([0, 3, 3, 5])),
        ByteMaskedArray(
            Index8(np.array([0, 1, 1, 0, 1])),
            NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
            True,
        ),
    )
    out = ak.drop_none(ak.Array(layout), axis=1)
    assert out.tolist() == [[2.2, 3.3], [], [5.5]]


def test_BitMaskedArray():
    layout = ListOffsetArray(
        Index64(np.array([0, 3, 3, 5])),
        BitMaskedArray(
            IndexU8(np.array([15])),
            NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
            True,
            5,
            True,
        ),
    )
    out = ak.drop_none(ak.Array(layout), axis=1)
    assert out.tolist() == [[1.1, 2.2, 3.3], [], [4.4]]
