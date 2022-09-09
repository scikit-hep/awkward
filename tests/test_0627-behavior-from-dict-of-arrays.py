# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    simple = {"what": "ever"}
    one = ak._v2.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], behavior=simple)
    two = ak._v2.Array([["one", "two"], ["three"], ["four", "five"]], behavior=simple)
    three = ak._v2.operations.cartesian({"one": one, "two": two})
    assert three.behavior == {"what": "ever"}
