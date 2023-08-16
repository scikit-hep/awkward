# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.zip({"x": [[1, 2, 3]], "y": [[4, 5]]}, depth_limit=1)
    sliced = array[..., np.newaxis]
    assert sliced.layout.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.RegularArray(
                    ak.contents.ListOffsetArray(
                        ak.index.Index64([0, 3]), ak.contents.NumpyArray([1, 2, 3])
                    ),
                    1,
                ),
                ak.contents.RegularArray(
                    ak.contents.ListOffsetArray(
                        ak.index.Index64([0, 2]), ak.contents.NumpyArray([4, 5])
                    ),
                    1,
                ),
            ],
            ["x", "y"],
        )
    )
