# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np # noqa: F401

to_list = ak._v2.operations.to_list


def test_lengths_empty_regular_slices():
    a = ak._v2.Array([[1, 2, 3], [4, 5, 6]])
    # assert to_list(a[:, []]) == [[], []]

    b = ak._v2.operations.to_regular(a, axis=1)
    # assert to_list(b[:, []]) == [[], []]


    d = np.arange(2 * 3).reshape(2, 3)
    e = ak._v2.contents.NumpyArray(d)
    # assert ak._v2.to_list(d[:,[]]) == ak._v2.to_list(b[:,[]]) == [[], []] 
    # assert ak._v2.to_list(d[1:,[]]) == ak._v2.to_list(e[1:,[]]) == [[]]
    # assert d[:,[]].shape == ak._v2.to_numpy(e[:,[]]).shape #(2, 0) == (0, 0, 2, 3)

    d = np.arange(5*7*11*13*17).reshape(5, 7, 11, 13, 17) 
    e = ak._v2.contents.NumpyArray(d)
    assert d[[],:,[]].shape == ak._v2.to_numpy(e[[],:,[]]).shape #(0, 7, 13, 17) == (0, 0, 5, 7, 11, 13, 17)
