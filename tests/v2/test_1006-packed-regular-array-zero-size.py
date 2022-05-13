# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test():
    array = ak._v2.contents.RegularArray(
        ak._v2.contents.NumpyArray(np.empty(0, dtype=np.int32)),
        size=0,
        zeros_length=1,
    )
    packed = ak._v2.operations.packed(array)
    assert to_list(packed) == [[]]
