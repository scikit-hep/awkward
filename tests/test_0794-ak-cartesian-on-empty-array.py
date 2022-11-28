# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    nparray = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.int64))
    listarray = ak.contents.ListArray(
        ak.index.Index64(np.array([0], dtype=np.int64)),
        ak.index.Index64(np.array([3], dtype=np.int64)),
        nparray,
    )
    indexedarray = ak.contents.IndexedArray(ak.index.Index64([]), listarray)

    cart = ak.operations.cartesian([indexedarray, indexedarray], nested=True)
    assert str(cart.type) == "0 * var * var * (int64, int64)"
    assert to_list(cart) == []

    cart = ak.operations.cartesian([indexedarray, indexedarray], nested=False)
    assert str(cart.type) == "0 * var * (int64, int64)"
    assert to_list(cart) == []
