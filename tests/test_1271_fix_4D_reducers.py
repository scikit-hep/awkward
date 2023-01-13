# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    akarray = ak.highlevel.Array(
        [[[[1], [4]], [[5], [8]]], [[[9], [12]], [[13], [16]]]]
    )
    nparray = np.array([[[[1], [4]], [[5], [8]]], [[[9], [12]], [[13], [16]]]])

    assert ak.sum(akarray, axis=3).to_list() == np.sum(nparray, axis=3).tolist()
    assert ak.sum(akarray, axis=2).to_list() == np.sum(nparray, axis=2).tolist()
    assert ak.sum(akarray, axis=1).to_list() == np.sum(nparray, axis=1).tolist()
    assert ak.sum(akarray, axis=0).to_list() == np.sum(nparray, axis=0).tolist()
