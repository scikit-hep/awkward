# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test_merge_option():
    x = ak.layout.IndexedArray64(
        ak.layout.Index64([0, 1]), ak.layout.NumpyArray([1, 2, 3])
    )
    y = ak.layout.IndexedOptionArray64(
        ak.layout.Index64([0, 1, -1]), ak.layout.NumpyArray([1, 2, 3])
    )

    assert isinstance(ak.concatenate((y, x)).layout, ak.layout.IndexedOptionArray64)
