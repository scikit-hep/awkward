# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    np_content = np.asfortranarray(np.arange(15).reshape(3, 5))
    ak_content = ak.layout.NumpyArray(np_content)
    offsets = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2, 3, 3]))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, ak_content)
    assert ak.to_list(listoffsetarray[1, 0]) == [0, 1, 2, 3, 4]
    assert ak.to_list(listoffsetarray[3, 0]) == [5, 6, 7, 8, 9]
