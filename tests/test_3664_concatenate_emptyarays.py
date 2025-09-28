# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_all_empty():
    ak.concatenate([ak.Array([None]) for _ in range(5000)], axis=0)
    ak.concatenate([ak.Array([{"x": None}]) for _ in range(5000)], axis=0)
    ak.concatenate([ak.Array([{"x": i, "y": None}]) for i in range(5000)], axis=0)


def test_empty_and_nonempty():
    arrays = []
    for i in range(5000):
        if np.random.choice([True, False]):
            arrays.append(ak.Array([None]))
        else:
            arrays.append(ak.Array([i]))
    ak.concatenate(arrays, axis=0)

    arrays = []
    for i in range(5000):
        if np.random.choice([True, False]):
            arrays.append(ak.Array([{"x": None}]))
        else:
            arrays.append(ak.Array([{"x": i}]))
    ak.concatenate(arrays, axis=0)

    arrays = []
    for i in range(5000):
        if np.random.choice([True, False]):
            arrays.append(ak.Array([{"x": i, "y": None}]))
        else:
            arrays.append(ak.Array([{"x": i, "y": i}]))
    ak.concatenate(arrays, axis=0)
