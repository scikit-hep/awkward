# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert [x.tolist() for x in ak.broadcast_arrays([1, 2, 3], [[], [1], [1, 2]])] == [
        [[], [2], [3, 3]],
        [[], [1], [1, 2]],
    ]

    assert [
        x.tolist()
        for x in ak.broadcast_arrays([1, 2, 3], [[], [{"x": 1}], [{"x": 1}, {"x": 2}]])
    ] == [
        [[], [{"x": 2}], [{"x": 3}, {"x": 3}]],
        [[], [{"x": 1}], [{"x": 1}, {"x": 2}]],
    ]

    assert [
        x.tolist()
        for x in ak.broadcast_arrays(
            [1, 2, 3],
            [[], [{"x": 1}], [{"x": 1}, {"x": 2}]],
            promote_scalar_to_record=False,
        )
    ] == [
        [[], [2], [3, 3]],
        [[], [{"x": 1}], [{"x": 1}, {"x": 2}]],
    ]

    assert [
        x.tolist()
        for x in ak.broadcast_arrays(
            [{"y": 1}, {"y": 2}, {"y": 3}], [[], [{"x": 1}], [{"x": 1}, {"x": 2}]]
        )
    ] == [
        [[], [{"y": 2}], [{"y": 3}, {"y": 3}]],
        [[], [{"x": 1}], [{"x": 1}, {"x": 2}]],
    ]
