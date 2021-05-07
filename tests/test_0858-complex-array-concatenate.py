# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array([1j])
    assert array.tolist() == [1j]

    array1 = ak.concatenate(([3j], [3j]))
    assert array1.tolist() == [3j, 3j]
