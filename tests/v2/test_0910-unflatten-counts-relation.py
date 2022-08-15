# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    layout = ak._v2.contents.IndexedArray(
        ak._v2.index.Index64(np.array([3, 1, 0, 2])),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 3, 6, 9, 12])),
            ak._v2.contents.NumpyArray(np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])),
        ),
    )

    assert ak._v2.operations.unflatten(
        layout,
        ak._v2.operations.flatten(ak._v2.operations.run_lengths(layout)),
        axis=1,
    ).tolist() == [[[3, 3, 3]], [[1, 1, 1]], [[0, 0, 0]], [[2, 2], [3]]]
