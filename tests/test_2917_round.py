# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward.operations import to_list


def test_round():
    """
    Testing Awkward implementations (cpu backend) of real, imag, and angle
    """
    arr_1 = ak.Array(
        [[1 + 0.111j, 2 + 0.222j, 3 + 0.333j], [], [4 + 0.444j, 5 + 0.555j]]
    )
    arr_2 = ak.Array([-2.5, -1.5, -0.6, -0.5, -0.4, 0.4, 0.5, 0.6, 1.5, 2.5])

    assert to_list(np.round(arr_1)) == [
        [(1 + 0j), (2 + 0j), (3 + 0j)],
        [],
        [(4 + 0j), (5 + 1j)],
    ]

    assert to_list(np.round(arr_1, decimals=2)) == [
        [(1 + 0.11j), (2 + 0.22j), (3 + 0.33j)],
        [],
        [(4 + 0.44j), (5 + 0.56j)],
    ]

    assert to_list(np.round(arr_2)) == [
        -2.0,
        -2.0,
        -1.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        2.0,
    ]
