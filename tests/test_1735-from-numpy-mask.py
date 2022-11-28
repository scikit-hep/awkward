# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    np_array = np.ma.MaskedArray(
        [[1, 2, 3], [4, 5, 6]], mask=[[False, True, False], [True, True, False]]
    )
    ak_array = ak.from_numpy(np_array)

    assert ak_array.to_list() == [[1, None, 3], [None, None, 6]]
