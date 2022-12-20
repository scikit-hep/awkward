# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    left = ak.contents.NumpyArray(
        np.arange(10), parameters={"left": "leftie", "some": "A"}
    )
    right = ak.contents.NumpyArray(
        np.arange(10), parameters={"right": "rightie", "some": "B"}
    )
    array = ak.contents.UnionArray.simplified(
        ak.index.Index8(np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int8)),
        ak.index.Index32(np.array([0, 1, 2, 3, 0, 1, 2, 3], np.int32)),
        [left, right],
        parameters={"some": "other"},
    )
    assert array.parameter("some") == "other"
    assert array.parameter("left") == "leftie"
    assert array.parameter("right") == "rightie"
