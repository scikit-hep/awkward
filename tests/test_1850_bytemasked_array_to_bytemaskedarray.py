# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.contents.NumpyArray(np.arange(5)),
        valid_when=True,
    )
    result = layout.to_ByteMaskedArray(False)
    assert layout.to_list() == [None, 1, None, 3, None]
    assert result.to_list() == [None, 1, None, 3, None]
    assert layout.backend.nplike.asarray(result.mask.data).tolist() == [
        1,
        0,
        1,
        0,
        1,
    ]

    # Check this works
    layout.to_typetracer().to_ByteMaskedArray(False)
