# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    akarray = ak._v2.highlevel.Array(
        [[[[1], [4]], [[5], [8]]], [[[9], [12]], [[13], [16]]]]
    )
    nparray = np.array([[[[1], [4]], [[5], [8]]], [[[9], [12]], [[13], [16]]]])

    assert ak._v2.sum(akarray, axis=3).tolist() == np.sum(nparray, axis=3).tolist()
    assert ak._v2.sum(akarray, axis=2).tolist() == np.sum(nparray, axis=2).tolist()
    assert ak._v2.sum(akarray, axis=1).tolist() == np.sum(nparray, axis=1).tolist()
    assert ak._v2.sum(akarray, axis=0).tolist() == np.sum(nparray, axis=0).tolist()
