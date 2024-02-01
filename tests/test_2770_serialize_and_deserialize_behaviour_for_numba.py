# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")


@numba.njit
def func(array):
    return array


def test_ArrayBuilder_behavior():
    SOME_ATTRS = {"FOO": "BAR"}
    builder = ak.ArrayBuilder(behavior=SOME_ATTRS)

    assert builder.behavior is SOME_ATTRS
    assert func(builder).behavior == SOME_ATTRS


def test_ArrayBuilder_non_picklable_behavior():
    def make_add_xyr():
        def add_xyr(left, right):
            x = left.x + right.x
            y = left.y + right.y
            return ak.zip(
                {
                    "x": x,
                    "y": y,
                    "r": np.sqrt(x**2 + y**2),
                },
                with_name="xyr",
            )

        return add_xyr

    behavior = {(np.add, "xyr", "xyr"): make_add_xyr()}
    builder = ak.ArrayBuilder(behavior=behavior)
    result = func(builder)
    assert result.behavior.keys() == behavior.keys()
