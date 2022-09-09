# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test():
    nparray = ak._v2.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64))
    listarray = ak._v2.contents.ListArray(
        ak._v2.index.Index64(np.array([0], dtype=np.int64)),
        ak._v2.index.Index64(np.array([3], dtype=np.int64)),
        nparray,
    )
    indexedarray = ak._v2.contents.IndexedArray(ak._v2.index.Index64([]), listarray)

    cart = ak._v2.operations.cartesian([indexedarray, indexedarray], nested=True)
    assert str(cart.type) == "0 * var * var * (int64, int64)"
    assert to_list(cart) == []

    cart = ak._v2.operations.cartesian([indexedarray, indexedarray], nested=False)
    assert str(cart.type) == "0 * var * (int64, int64)"
    assert to_list(cart) == []
