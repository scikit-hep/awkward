# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([1, 2, 3, 4])
    first = ak.firsts(array, axis=0)
    assert first == 1


def test_non_scalar():
    array = ak.Array([[1, 2, 3], [4]])
    first = ak.firsts(array, axis=0)
    assert first.to_list() == [1, 2, 3]
