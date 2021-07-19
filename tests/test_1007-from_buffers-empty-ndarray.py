# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.from_numpy(np.zeros((3, 0), dtype=np.int32))
    buffs = ak.to_buffers(array)
    new_array = ak.from_buffers(*buffs)

    assert ak.to_list(new_array) == [[], [], []]
