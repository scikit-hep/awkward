# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([None, [1, 2, 3, [None, {"x": [None, 2], "y": [1, 4]}]]])

    assert ak.fill_none(array, -1.0, axis=None).to_list() == [
        -1.0,
        [1, 2, 3, [-1.0, {"x": [-1.0, 2], "y": [1, 4]}]],
    ]
