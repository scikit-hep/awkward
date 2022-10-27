# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    array = np.random.random(size=512).astype(dtype=np.float64)
    assert ak.type(array) == ak.types.ArrayType(ak.types.NumpyType("float64"), 512)
