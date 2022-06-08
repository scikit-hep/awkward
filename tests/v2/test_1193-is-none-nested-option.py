# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    index_of_index = ak._v2.highlevel.Array(
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(np.r_[0, 1, 2, -1]),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index64(np.r_[0, -1, 2, 3]),
                ak._v2.contents.NumpyArray(np.r_[1, 2, 3, 4]),
            ),
        )
    )

    mask = ak._v2.operations.is_none(index_of_index)
    assert mask.tolist() == [False, True, False, True]
