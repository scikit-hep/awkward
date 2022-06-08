# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    layout = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 1], dtype=np.int64)),
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64)),
            ak._v2.contents.RegularArray(
                ak._v2.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4])), 4
            ),
        ),
    )
    array = ak._v2.highlevel.Array(layout)

    assert (
        str(ak._v2.operations.min(array, axis=-1, mask_identity=False).type)
        == "1 * var * float64"
    )
