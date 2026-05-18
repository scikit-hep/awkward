from __future__ import annotations

import awkward as ak


def test():
    array = ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])

    assert ak.combinations(array, 2, axis=-1).tolist() == [
        [(0.0, 1.1), (0.0, 2.2), (1.1, 2.2)],
        [],
        [(3.3, 4.4)],
        [],
        [(6.6, 7.7), (6.6, 8.8), (6.6, 9.9), (7.7, 8.8), (7.7, 9.9), (8.8, 9.9)],
    ]
    assert ak.argcombinations(array, 2, axis=-1).tolist() == [
        [(0, 1), (0, 2), (1, 2)],
        [],
        [(0, 1)],
        [],
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    ]
