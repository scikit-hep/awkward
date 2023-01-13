# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(
                np.array(
                    [0, 2, 4, 4],
                    dtype=np.int_,
                )
            ),
            ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(4 * 2)), size=2),
        )
    )

    assert ak.sum(array, axis=1).to_list() == [[2, 4], [10, 12], [0, 0]]
