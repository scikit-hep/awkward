# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_equal_union():
    union_1 = ak.from_iter([1, None, {"x": 2}], highlevel=False)
    union_2 = ak.from_iter([1, None, {"x": 2}], highlevel=False)

    assert union_1.is_equal_to(union_2)


def test_unequal_union():
    union_1 = ak.from_iter([1, None, {"x": 2}, 3], highlevel=False)
    union_2 = ak.from_iter([1, None, {"x": 2}, 2], highlevel=False)

    assert not union_1.is_equal_to(union_2)
