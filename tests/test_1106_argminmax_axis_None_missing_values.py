# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    x = ak.highlevel.Array([1, 2, 3, None, 4])
    assert ak.operations.argmax(x) == 4
