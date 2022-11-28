# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.operations.values_astype(
        ak.highlevel.Array([1.1, 2.2, None, 3.3]), np.float32
    )
    assert str(ak.operations.fill_none(array, np.float32(0)).type) == "4 * float32"
