# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.RecordArray(
        [
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 3]),
                ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.uint16)),
            ),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0, 2]),
                ak.contents.NumpyArray(np.array([4, 5], dtype=np.uint16)),
            ),
        ],
        ["x", "y"],
    )
    sliced = layout[..., np.newaxis]
    assert sliced.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.RegularArray(
                    ak.contents.ListOffsetArray(
                        ak.index.Index64([0, 3]),
                        ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.uint16)),
                    ),
                    1,
                ),
                ak.contents.RegularArray(
                    ak.contents.ListOffsetArray(
                        ak.index.Index64([0, 2]),
                        ak.contents.NumpyArray(np.array([4, 5], dtype=np.uint16)),
                    ),
                    1,
                ),
            ],
            ["x", "y"],
        )
    )
