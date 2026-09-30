# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import awkward as ak


def test_noop():
    assert ak.categories(ak.Array([1])) is None
