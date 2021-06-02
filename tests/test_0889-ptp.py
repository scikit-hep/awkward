# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_regular():
    np_data = np.array(
        [[1, 3, 5, 4, 2], [3, 7, 8, 2, 4], [2, 3, 1, 7, 7], [5, 1, 9, 10, 2]]
    )
    ak_data = ak.from_numpy(np_data)

    assert ak.ptp(ak_data, axis=1).tolist() == np.ptp(np_data, axis=1).tolist()
    assert ak.ptp(ak_data, axis=0).tolist() == np.ptp(np_data, axis=0).tolist()
    assert ak.ptp(ak_data) == np.ptp(np_data)


def test_jagged():
    data = ak.Array(
        [
            [1, 3, 5, 4, 2],
            [],
            [2, 3, 1],
            [5],
        ]
    )
    assert ak.ptp(data, axis=1, mask_identity=False).tolist() == [4, 0, 2, 0]
    assert ak.ptp(data, axis=1).tolist() == [4, None, 2, 0]
    assert ak.ptp(data, axis=0).tolist() == [4, 0, 4, 0, 0]
    assert ak.ptp(data) == 4
