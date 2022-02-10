# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    index_of_index = ak.Array(
        ak.layout.IndexedOptionArray64(
            ak.layout.Index64(np.r_[0, 1, 2, -1]),
            ak.layout.IndexedOptionArray64(
                ak.layout.Index64(np.r_[0, -1, 2, 3]),
                ak.layout.NumpyArray(np.r_[1, 2, 3, 4]),
            ),
        )
    )

    mask = ak.is_none(index_of_index)
    assert ak.to_list(mask) == [False, True, False, True]
