# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.typetracer import is_unknown_length


def test():
    array = ak.Array([1, 2, None])
    result = ak.fill_none(array, 0)
    assert result.to_list() == [1, 2, 0]

    array_tt = ak.Array(array.layout.to_typetracer(forget_length=True))
    result_tt = ak.fill_none(array_tt, 0)
    assert is_unknown_length(result_tt.layout.length)
