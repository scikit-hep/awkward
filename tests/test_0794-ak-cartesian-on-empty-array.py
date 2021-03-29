# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    nparray = ak.layout.NumpyArray(np.array([1, 2, 3], dtype=np.int64))
    listarray = ak.layout.ListArray64(
        ak.layout.Index64(np.array([0], dtype=np.int64)),
        ak.layout.Index64(np.array([3], dtype=np.int64)),
        nparray,
    )
    indexedarray = ak.layout.IndexedArray64(ak.layout.Index64([]), listarray)

    cart = ak.cartesian([indexedarray, indexedarray], nested=True)
    assert str(cart.type) == "0 * var * var * (int64, int64)"
    assert ak.to_list(cart) == []

    cart = ak.cartesian([indexedarray, indexedarray], nested=False)
    assert str(cart.type) == "0 * var * (int64, int64)"
    assert ak.to_list(cart) == []
