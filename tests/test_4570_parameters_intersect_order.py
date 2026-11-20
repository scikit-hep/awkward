# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak
from awkward._parameters import parameters_intersect

KEYS = [f"key{i}" for i in range(20)]


def test_parameters_intersect_keeps_the_left_order():
    left = {key: i for i, key in enumerate(KEYS)}
    right = dict(reversed(list(left.items())))
    assert list(parameters_intersect(left, right)) == KEYS


def test_concatenate_keeps_the_parameter_order():
    a = ak.Array([[1.0], [2.0]])
    for key in KEYS:
        a = ak.with_parameter(a, key, key)
    assert list(ak.concatenate([a, a]).layout.parameters) == KEYS
