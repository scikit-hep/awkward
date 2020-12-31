# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    simple = {"what": "ever"}
    one = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], behavior=simple)
    two = ak.Array([["one", "two"], ["three"], ["four", "five"]], behavior=simple)
    three = ak.cartesian({"one": one, "two": two})
    assert three.behavior == {"what": "ever"}
