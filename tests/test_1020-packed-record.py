# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.zip({"x": np.arange(512), "y": np.arange(512)})

    record = array[10]
    packed = ak.packed(record)
    assert len(packed.layout.array) == 1
    assert ak.to_list(packed.layout.array.contents[0]) == [10]
    assert ak.to_list(packed.layout.array.contents[1]) == [10]
