# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
