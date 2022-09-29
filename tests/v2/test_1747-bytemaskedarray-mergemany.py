# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    x = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.r_[0, 1, 1]),
        ak._v2.contents.NumpyArray(np.arange(12)),
        valid_when=True,
    )
    y = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.r_[1, 1, 1, 0, 0]),
        ak._v2.contents.NumpyArray(np.arange(12)),
        valid_when=True,
    )
    z = x.merge(y)

    assert z.to_list() == [None, 1, 2, 0, 1, 2, None, None]
