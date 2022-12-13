# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_regular():
    np_data = np.array(
        [[1, 3, 5, 4, 2], [3, 7, 8, 2, 4], [2, 3, 1, 7, 7], [5, 1, 9, 10, 2]]
    )
    ak_data = ak.operations.from_numpy(np_data)

    assert (
        ak.operations.ptp(ak_data, axis=1).to_list() == np.ptp(np_data, axis=1).tolist()
    )
    assert (
        ak.operations.ptp(ak_data, axis=0).to_list() == np.ptp(np_data, axis=0).tolist()
    )
    assert ak.operations.ptp(ak_data) == np.ptp(np_data)


def test_jagged():
    data = ak.highlevel.Array(
        [
            [1, 3, 5, 4, 2],
            [],
            [2, 3, 1],
            [5],
        ]
    )
    assert ak.operations.ptp(data, axis=1, mask_identity=False).to_list() == [
        4,
        0,
        2,
        0,
    ]
    assert ak.operations.ptp(data, axis=1).to_list() == [4, None, 2, 0]
    assert ak.operations.ptp(data, axis=0).to_list() == [4, 0, 4, 0, 0]
    assert ak.operations.ptp(data) == 4
