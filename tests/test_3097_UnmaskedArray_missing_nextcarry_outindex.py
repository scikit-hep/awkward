# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray, UnmaskedArray
from awkward.index import Index64


def test():
    layout = ListOffsetArray(
        Index64(np.array([0, 3, 3, 5])),
        UnmaskedArray(NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))),
    )
    out = ak.drop_none(ak.Array(layout), axis=1)
    assert out.tolist() == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
