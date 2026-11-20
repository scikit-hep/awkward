# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os
import subprocess
import sys

import awkward as ak
from awkward._parameters import parameters_intersect

KEYS = [f"key{i}" for i in range(20)]


def test_parameters_intersect_keeps_the_left_order():
    left = {key: i for i, key in enumerate(KEYS)}
    right = dict(reversed(list(left.items())))
    assert list(parameters_intersect(left, right)) == KEYS


def test_concatenate_parameters_do_not_depend_on_the_hash_seed():
    code = (
        "import awkward as ak\n"
        "a = ak.Array([[1.0], [2.0]])\n"
        f"for key in {KEYS!r}:\n"
        "    a = ak.with_parameter(a, key, key)\n"
        "print(list(ak.concatenate([a, a]).layout.parameters))\n"
    )
    orders = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in range(5)
    }
    assert len(orders) == 1
    assert orders.pop().strip() == repr(KEYS)


def test_unequal_values_are_still_dropped():
    out = ak.concatenate(
        [
            ak.with_parameter(ak.Array([1]), "x", 1),
            ak.with_parameter(ak.Array([2]), "x", 2),
        ]
    )
    assert "x" not in (out.layout.parameters or {})
