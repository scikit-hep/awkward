# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test():
    array = ak.Array(
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4]], [], [[5.5], [], [6.6, 7.7, 8.8, 9.9]]]
    )
    assert ak.to_list(ak.local_index(array, axis=0)) == [0, 1, 2, 3]
    assert ak.to_list(ak.local_index(array, axis=1)) == [[0, 1], [0], [], [0, 1, 2]]
    assert ak.to_list(ak.local_index(array, axis=2)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert ak.to_list(ak.local_index(array, axis=-1)) == [
        [[0, 1, 2], []],
        [[0, 1]],
        [],
        [[0], [], [0, 1, 2, 3]],
    ]
    assert ak.to_list(ak.local_index(array, axis=-2)) == [[0, 1], [0], [], [0, 1, 2]]
    assert ak.to_list(ak.local_index(array, axis=-3)) == [0, 1, 2, 3]

    assert ak.to_list(
        ak.zip(
            [
                ak.local_index(array, axis=0),
                ak.local_index(array, axis=1),
                ak.local_index(array, axis=2),
            ]
        )
    ) == [
        [[(0, 0, 0), (0, 0, 1), (0, 0, 2)], []],
        [[(1, 0, 0), (1, 0, 1)]],
        [],
        [[(3, 0, 0)], [], [(3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3)]],
    ]
