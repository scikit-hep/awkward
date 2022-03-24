# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    ak_array = ak.Array([1, 2, 3])
    assert ak.singletons(ak_array).tolist() == [[1], [2], [3]]
