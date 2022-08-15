# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    x = ak._v2.highlevel.Array([1, 2, 3, None, 4])
    assert ak._v2.operations.argmax(x) == 4
