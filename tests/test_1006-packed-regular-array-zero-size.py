# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    array = ak.contents.RegularArray(
        ak.contents.NumpyArray(np.empty(0, dtype=np.int32)),
        size=0,
        zeros_length=1,
    )
    packed = ak.operations.to_packed(array)
    assert to_list(packed) == [[]]
