# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(
                np.array(
                    [0, 2, 4, 4],
                    dtype=np.int_,
                )
            ),
            ak.layout.RegularArray(ak.layout.NumpyArray(np.arange(4 * 2)), size=2),
        )
    )

    assert ak.sum(array, axis=1).to_list() == [[2, 4], [10, 12], [0, 0]]
