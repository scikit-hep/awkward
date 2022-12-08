# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.array([3, 1, 0, 2])),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 6, 9, 12])),
            ak.contents.NumpyArray(np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3])),
        ),
    )

    assert ak.operations.unflatten(
        layout,
        ak.operations.flatten(ak.operations.run_lengths(layout)),
        axis=1,
    ).to_list() == [[[3, 3, 3]], [[1, 1, 1]], [[0, 0, 0]], [[2, 2], [3]]]
