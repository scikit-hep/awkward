# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array(["123", "45", "00001", "    0000   ", "-2.6", "-3", "1e1", "1E1"])
    assert ak.strings_astype(array, np.float64).tolist() == [
        123,
        45,
        1,
        0,
        -2.6,
        -3,
        10,
        10,
    ]
    assert ak.strings_astype(
        array[[True, True, True, True, False, True, False, False]], np.int64
    ).tolist() == [123, 45, 1, 0, -3]

    array = ak.Array(
        [["123", "45", "00001"], [], ["    0000   ", "-2.6"], ["-3"], ["1e1", "1E1"]]
    )
    assert ak.strings_astype(array, np.float64).tolist() == [
        [123, 45, 1],
        [],
        [0, -2.6],
        [-3],
        [10, 10],
    ]
