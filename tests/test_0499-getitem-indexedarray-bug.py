# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test():
    one_content = ak.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9, 10.0]]
    ).layout
    one_starts = ak.layout.Index64(np.array([0, 2, 3, 3], dtype=np.int64))
    one_stops = ak.layout.Index64(np.array([2, 3, 3, 5], dtype=np.int64))
    one = ak.layout.ListArray64(one_starts, one_stops, one_content)
    assert ak.to_list(one) == [
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9, 10.0]],
    ]

    two_content = ak.Array(
        [
            [123],
            [1.1, 2.2, 3.3],
            [],
            [234],
            [4.4, 5.5],
            [345],
            [6.6],
            [7.7, 8.8, 9.9, 10.0],
            [456],
        ]
    ).layout
    two_starts = ak.layout.Index64(np.array([1, 4, 5, 6], dtype=np.int64))
    two_stops = ak.layout.Index64(np.array([3, 5, 5, 8], dtype=np.int64))
    two = ak.layout.ListArray64(two_starts, two_stops, two_content)
    assert ak.to_list(two) == [
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9, 10.0]],
    ]

    assert ak.to_list(one[[[[0, 1, 2], []], [[0, 1]], [], [[0], [0, 1, 2, 3]]]]) == [
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9, 10.0]],
    ]
    assert ak.to_list(two[[[[0, 1, 2], []], [[0, 1]], [], [[0], [0, 1, 2, 3]]]]) == [
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9, 10.0]],
    ]
