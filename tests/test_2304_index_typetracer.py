# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test():
    array = ak.Array([[1, 2, 3], [4, 5, 6]], backend="typetracer")

    with pytest.raises(IndexError):
        array[0, 0, 0]

    with pytest.raises(IndexError):
        array[0, :, 0]

    with pytest.raises(IndexError):
        array[0, :, ..., 0]

    with pytest.raises(IndexError):
        array[0, [0], [0]]
