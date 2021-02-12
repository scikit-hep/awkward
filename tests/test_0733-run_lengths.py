# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.Array([3, 3, 3, 5, 5, 9, 9, 9, 9, 1, 3, 3])
    assert ak.run_lengths(array).tolist() == [3, 2, 4, 1, 2]

    array = ak.Array([[3, 3, 3, 5], [5], [], [9, 9], [9, 9], [1, 3, 3]])
    assert ak.run_lengths(array).tolist() == [[3, 1], [1], [], [2], [2], [1, 2]]

    array = ak.repartition(ak.Array([3, 3, 3, 5, 5, 9, 9, 9, 9, 1, 3, 3]), 7)
    assert ak.run_lengths(array).tolist() == [3, 2, 4, 1, 2]

    array = ak.repartition(
        ak.Array([[3, 3, 3, 5], [5], [], [9, 9], [9, 9], [1, 3, 3]]), 4
    )
    assert ak.run_lengths(array).tolist() == [[3, 1], [1], [], [2], [2], [1, 2]]


def test_groupby():
    array = ak.Array(
        [
            {"x": 1, "y": 1.1},
            {"x": 2, "y": 2.2},
            {"x": 1, "y": 1.1},
            {"x": 3, "y": 3.3},
            {"x": 1, "y": 1.1},
            {"x": 2, "y": 2.2},
        ]
    )
    sorted = array[ak.argsort(array.x)]
    assert sorted.x.tolist() == [1, 1, 1, 2, 2, 3]
    assert ak.run_lengths(sorted.x).tolist() == [3, 2, 1]
    assert ak.unflatten(sorted, ak.run_lengths(sorted.x)).tolist() == [
        [{"x": 1, "y": 1.1}, {"x": 1, "y": 1.1}, {"x": 1, "y": 1.1}],
        [{"x": 2, "y": 2.2}, {"x": 2, "y": 2.2}],
        [{"x": 3, "y": 3.3}],
    ]

    array = ak.Array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 1, "y": 1.1}],
            [{"x": 3, "y": 3.3}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        ]
    )
    sorted = array[ak.argsort(array.x)]
    assert sorted.x.tolist() == [[1, 1, 2], [1, 2, 3]]
    assert ak.run_lengths(sorted.x).tolist() == [[2, 1], [1, 1, 1]]
    counts = ak.flatten(ak.run_lengths(sorted.x), axis=None)
    assert ak.unflatten(sorted, counts, axis=-1).tolist() == [
        [[{"x": 1, "y": 1.1}, {"x": 1, "y": 1.1}], [{"x": 2, "y": 2.2}]],
        [[{"x": 1, "y": 1.1}], [{"x": 2, "y": 2.2}], [{"x": 3, "y": 3.3}]],
    ]


def test_onstrings1():
    data = ak.Array(["one", "one", "one", "two", "two", "three", "two", "two"])
    assert ak.run_lengths(data).tolist() == [3, 2, 1, 2]


def test_onstrings2():
    data = ak.Array([["one", "one"], ["one", "two", "two"], ["three", "two", "two"]])
    assert ak.run_lengths(data).tolist() == [[2], [1, 2], [1, 2]]
