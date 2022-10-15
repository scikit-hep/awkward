import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401


def test():
    akarray = ak.highlevel.Array(
        [
            [[[1, 2, 7], [4, 3, 9]], [[5, 0, 6], [8, 8, 3]]],
            [[[9, 5, 6], [12, 31, 55]], [[1, 73, 4], [10, 20, 106]]],
        ]
    )
    nparray = ak.to_numpy(akarray)

    assert ak.cumsum(akarray, axis=3).tolist() == np.sum(nparray, axis=3).tolist()
    assert ak.cumsum(akarray, axis=2).tolist() == np.sum(nparray, axis=2).tolist()
    assert ak.cumsum(akarray, axis=1).tolist() == np.sum(nparray, axis=1).tolist()
    assert ak.cumsum(akarray, axis=0).tolist() == np.sum(nparray, axis=0).tolist()


def test_nones():
    akarray = ak.highlevel.Array(
        [
            [[[1, 2, 7], [4, 3, 9]], [[5, 0, None], [8, 8, 3]]],
            [[[9, None, 6], [12, 31, 55]], [[None, 73, 4], [10, 20, 106]]],
        ]
    )
    nparray = ak.to_numpy(akarray)

    assert ak.cumsum(akarray, axis=3).tolist() == np.sum(nparray, axis=3).tolist()
    assert ak.cumsum(akarray, axis=2).tolist() == np.sum(nparray, axis=2).tolist()
    assert ak.cumsum(akarray, axis=1).tolist() == np.sum(nparray, axis=1).tolist()
    assert ak.cumsum(akarray, axis=0).tolist() == np.sum(nparray, axis=0).tolist()
