# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    layout = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([3, 1, 0, 2])),
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([0, 3, 6, 9, 12])),
            ak.layout.NumpyArray(np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])),
        ),
    )

    assert ak.unflatten(
        layout, ak.flatten(ak.run_lengths(layout)), axis=1
    ).tolist() == [[[3, 3, 3]], [[1, 1, 1]], [[0, 0, 0]], [[2, 2], [3]]]
