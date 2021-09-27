# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    layout = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([0, 1], dtype=np.int64)),
        ak.layout.IndexedArray64(
            ak.layout.Index64(np.array([0, 1, 2, 3], dtype=np.int64)),
            ak.layout.RegularArray(
                ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4])), 4
            ),
        ),
    )
    array = ak.Array(layout)

    assert str(ak.min(array, axis=-1, mask_identity=False).type) == "1 * var * float64"
