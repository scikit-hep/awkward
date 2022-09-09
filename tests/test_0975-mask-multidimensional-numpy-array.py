# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test():
    array = ak._v2.highlevel.Array(
        ak._v2.contents.RegularArray(
            ak._v2.contents.NumpyArray(np.r_[1, 2, 3, 4, 5, 6, 7, 8, 9]), 3
        )
    )
    mask = ak._v2.highlevel.Array(
        ak._v2.contents.NumpyArray(
            np.array([[True, True, True], [True, True, False], [True, False, True]])
        )
    )

    assert ak._v2.operations.mask(array, mask).to_list() == [
        [1, 2, 3],
        [4, 5, None],
        [7, None, 9],
    ]
