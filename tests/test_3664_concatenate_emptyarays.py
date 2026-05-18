# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import sys

import numpy as np

import awkward as ak


def test_all_empty():
    result = ak.concatenate([ak.Array([None]) for _ in range(5000)], axis=0)
    expected = ak.Array([None] * 5000)
    assert ak.array_equal(result, expected)

    result = ak.concatenate([ak.Array([{"x": None}]) for _ in range(5000)], axis=0)
    expected = ak.Array([{"x": None}] * 5000)
    assert ak.array_equal(result, expected)

    result = ak.concatenate(
        [ak.Array([{"x": i, "y": None}]) for i in range(5000)], axis=0
    )
    expected = ak.Array([{"x": i, "y": None} for i in range(5000)])
    assert ak.array_equal(result, expected)


def test_empty_and_nonempty():
    N = sys.getrecursionlimit()
    rng = np.random.default_rng(42)
    choices1 = np.concatenate(
        (np.array([True, True]), rng.choice([True, False], size=N))
    )
    choices2 = np.concatenate(
        (np.array([False, False]), rng.choice([True, False], size=N))
    )
    choices3 = np.concatenate(
        (np.array([True, False]), rng.choice([True, False], size=N))
    )
    choices4 = np.concatenate(
        (np.array([False, True]), rng.choice([True, False], size=N))
    )
    all_choices = [choices1, choices2, choices3, choices4]

    for choices in all_choices:
        rows = []
        for i in range(N):
            if choices[i]:
                rows.append(None)
            else:
                rows.append(i)
        result = ak.concatenate([ak.Array([row]) for row in rows], axis=0)
        expected = ak.Array(rows)
        assert ak.array_equal(result, expected)

        rows = []
        for i in range(N):
            if choices[i]:
                rows.append({"x": None})
            else:
                rows.append({"x": i})
        result = ak.concatenate([ak.Array([row]) for row in rows], axis=0)
        expected = ak.Array(rows)
        assert ak.array_equal(result, expected)

        rows = []
        for i in range(N):
            if choices[i]:
                rows.append({"x": i, "y": None})
            else:
                rows.append({"x": i, "y": i})
        result = ak.concatenate([ak.Array([row]) for row in rows], axis=0)
        expected = ak.Array(rows)
        assert ak.array_equal(result, expected)
