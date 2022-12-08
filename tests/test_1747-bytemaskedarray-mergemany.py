# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    x = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.r_[0, 1, 1]),
        ak.contents.NumpyArray(np.arange(12)),
        valid_when=True,
    )
    y = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.r_[1, 1, 1, 0, 0]),
        ak.contents.NumpyArray(np.arange(12)),
        valid_when=True,
    )
    z = x._mergemany([y])

    assert z.to_list() == [None, 1, 2, 0, 1, 2, None, None]
