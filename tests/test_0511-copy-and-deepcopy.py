# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import copy

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    np_array = np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    one = ak.Array(np_array)

    np_array[1] = 999
    assert ak.to_list(one) == [0.0, 999, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    two = copy.copy(one)
    np_array[3] = 123
    assert ak.to_list(two) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    three = copy.deepcopy(two)
    four = np.copy(two)
    np_array[5] = 321
    assert ak.to_list(three) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert ak.to_list(four) == [0.0, 999, 2.2, 123, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]

    assert ak.to_list(copy.deepcopy(ak.Array([[1, 2, 3], [], [4, 5]]))) == [
        [1, 2, 3],
        [],
        [4, 5],
    ]

    assert ak.to_list(copy.deepcopy(ak.Record({"one": 1, "two": 2.2}))) == ak.to_list(
        copy.deepcopy(ak.Record({"one": 1, "two": 2.2}))
    )
