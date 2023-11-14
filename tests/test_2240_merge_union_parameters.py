# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401

import awkward as ak


def test():
    one = ak.with_parameter([1, 2, [], [3, 4]], "one", "one")
    two = ak.with_parameter([100, 200, 300], "two", "two")
    three = ak.with_parameter([{"x": 1}, {"x": 2}, 5, 6, 7], "two", "two")

    # No parameter unions should occur here
    result = ak.concatenate((two, one, three))
    assert ak.parameters(result) == {}
