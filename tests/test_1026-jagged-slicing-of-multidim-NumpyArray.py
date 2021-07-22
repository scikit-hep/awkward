# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.r_[0, 4, 4, 6]),
            ak.layout.NumpyArray(np.arange(4 * 6, dtype=np.int64).reshape(6, 4)),
        )
    )
    ix = ak.Array([[[0], [1], [2], [3]], [], [[3], [2]]])
    assert array[ix].tolist() == [[[0], [5], [10], [15]], [], [[19], [22]]]
    assert str(array[ix].type) == "3 * var * var * int64"
