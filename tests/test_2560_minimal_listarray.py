# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length


def test():
    layout = ak.to_layout([[1, 2, 3, 4], [5, 6, 7, 8], [4, 5, 6, 9]]).to_typetracer(
        forget_length=True
    )
    result = layout.to_RegularArray()
    assert result.size is unknown_length
    assert result.length is unknown_length
