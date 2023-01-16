# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    assert ak.type(np.int16()) == ak.types.ScalarType(ak.types.NumpyType("int16"))
    assert ak.type(np.uint32) == ak.types.ScalarType(ak.types.NumpyType("uint32"))
    assert ak.type(np.dtype("complex128")) == ak.types.ScalarType(
        ak.types.NumpyType("complex128")
    )
