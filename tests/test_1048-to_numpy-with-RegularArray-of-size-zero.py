# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    a = ak.layout.RegularArray(ak.layout.EmptyArray(), size=0, zeros_length=3)
    assert ak.to_numpy(a).shape == (3, 0)
    assert ak.to_numpy(ak.Array([[], [], []])).shape == (3, 0)
