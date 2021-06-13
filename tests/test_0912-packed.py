# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_indexed_numpy_array():
    index = ak.layout.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak.layout.NumpyArray(np.arange(10))
    layout = ak.layout.IndexedArray64(index, content)

    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(layout) == ak.to_list(packed)

    assert isinstance(packed, ak.layout.NumpyArray)
    assert len(packed) == len(index)
