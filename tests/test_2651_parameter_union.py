# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.IndexedArray(
        ak.index.Index64([0, 1, 2]),
        ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.uint32)),
        parameters={"foo": {"bar": "baz"}},
    )
    result = layout.project()
    assert result.is_equal_to(
        ak.contents.NumpyArray(
            np.array([1, 2, 3], dtype=np.uint32), parameters={"foo": {"bar": "baz"}}
        )
    )
