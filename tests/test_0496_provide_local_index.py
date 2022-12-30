# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test():
    array = ak.highlevel.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert to_list(ak.operations.local_index(array, axis=0)) == [
        0,
        1,
        2,
        3,
    ]
    assert to_list(ak.operations.local_index(array, axis=1)) == [
        [0, 1],
        [0],
        [],
        [0, 1, 2],
    ]
    assert to_list(ak.operations.local_index(array, axis=2)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert to_list(ak.operations.local_index(array, axis=-1)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert to_list(ak.operations.local_index(array, axis=-2)) == [
        [0, 1],
        [0],
        [],
        [0, 1, 2],
    ]
    assert to_list(ak.operations.local_index(array, axis=-3)) == [
        0,
        1,
        2,
        3,
    ]

    assert to_list(
        ak.operations.zip(
            [
                ak.operations.local_index(array, axis=0),
                ak.operations.local_index(array, axis=1),
                ak.operations.local_index(array, axis=2),
            ]
        )
    ) == [
        [[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []],
        [[(1, 0, 0), (1, 0, 1)]],
        [],
        [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]],
    ]
