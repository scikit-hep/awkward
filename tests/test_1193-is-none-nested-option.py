# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    index_of_index = ak.highlevel.Array(
        ak.contents.IndexedOptionArray.simplified(
            ak.index.Index64(np.r_[0, 1, 2, -1]),
            ak.contents.IndexedOptionArray(
                ak.index.Index64(np.r_[0, -1, 2, 3]),
                ak.contents.NumpyArray(np.r_[1, 2, 3, 4]),
            ),
        )
    )

    mask = ak.operations.is_none(index_of_index)
    assert mask.to_list() == [False, True, False, True]
