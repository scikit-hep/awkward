# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    array = ak.Array([1, 2, None])
    result = ak.fill_none(array, 0)
    assert result.tolist() == [1, 2, 0]

    array_tt = ak.Array(array.layout.typetracer.forget_length())
    result_tt = ak.fill_none(array_tt, 0)
    assert result_tt.layout.length is ak._typetracer.UnknownLength
