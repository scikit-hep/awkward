# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    x = ak.from_numpy(np.arange(3, dtype=np.int64))
    y = ak.unflatten(x, len(x))
    assert y.to_list() == [[0, 1, 2]]
    assert y.type == ak.types.ArrayType(
        ak.types.RegularType(ak.types.NumpyType("int64"), 3), 1
    )
