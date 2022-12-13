# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    array = ak.highlevel.Array(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9]), 3
        )
    )
    mask = ak.highlevel.Array(
        ak.contents.NumpyArray(
            np.array([[True, True, True], [True, True, False], [True, False, True]])
        )
    )

    assert ak.operations.mask(array, mask).to_list() == [
        [1, 2, 3],
        [4, 5, None],
        [7, None, 9],
    ]
