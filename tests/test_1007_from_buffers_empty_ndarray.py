# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    array = ak.contents.NumpyArray(np.zeros((3, 0), dtype=np.int32))
    buffs = ak.operations.to_buffers(array)
    new_array = ak.operations.from_buffers(*buffs)

    assert to_list(new_array) == [[], [], []]
