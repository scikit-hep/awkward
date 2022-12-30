# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.highlevel.Array(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9]), 3
        )
    )
    condition = ak.highlevel.Array(
        ak.contents.NumpyArray(
            np.array([[True, True, True], [True, True, False], [True, False, True]])
        )
    )

    assert ak.operations.where(condition == 2, array, 2 * array).to_list() == [
        [2, 4, 6],
        [8, 10, 12],
        [14, 16, 18],
    ]
