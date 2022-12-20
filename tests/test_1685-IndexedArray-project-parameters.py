# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.IndexedArray(
        ak.index.Index64(np.array([0, 1, 2, 3], np.int64)),
        ak.contents.NumpyArray(
            np.arange(10), parameters={"this": "that", "some": "hidden"}
        ),
        parameters={"some": "other"},
    )
    assert layout.project().parameter("some") == "other"
    assert layout.project().parameter("this") == "that"
