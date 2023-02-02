# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    a = ak.Array([])

    assert a.tolist() == []
    assert str(a.type) == "0 * unknown"

    b = ak.values_astype(a, np.uint16)

    assert b.tolist() == []
    assert str(b.type) == "0 * uint16"
