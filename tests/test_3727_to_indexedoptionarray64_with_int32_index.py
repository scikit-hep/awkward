# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test():
    index32 = ak.index.Index(np.array([0, 1, -1], dtype=np.int32))
    content = ak.contents.NumpyArray(np.array([10, 20, 30]))
    arr = ak.contents.IndexedOptionArray(index32, content)
    new_arr = arr.to_IndexedOptionArray64()

    assert arr.index.dtype == np.dtype("int32")
    assert new_arr.index.dtype == np.dtype("int64")
    assert arr.to_list() == new_arr.to_list()
