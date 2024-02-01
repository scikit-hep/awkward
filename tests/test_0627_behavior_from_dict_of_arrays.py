# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    simple = {"what": "ever"}
    one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], behavior=simple)
    two = ak.Array([["one", "two"], ["three"], ["four", "five"]], behavior=simple)
    three = ak.operations.cartesian({"one": one, "two": two})
    assert three.behavior == {"what": "ever"}
