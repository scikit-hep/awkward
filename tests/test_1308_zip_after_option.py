# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_all_options():
    one = ak.highlevel.Array([1, 2, None])
    two = ak.highlevel.Array([None, 5, None])
    result = ak.operations.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * ?(int64, int64)"
    assert result.to_list() == [None, (2, 5), None]


def test_mixed_options():
    one = ak.highlevel.Array([1, 2, None])
    two = ak.highlevel.Array([4, 5, 6])
    result = ak.operations.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * ?(int64, int64)"
    assert result.to_list() == [(1, 4), (2, 5), None]


def test_no_options():
    one = ak.highlevel.Array([1, 2, 3])
    two = ak.highlevel.Array([4, 5, 6])
    result = ak.operations.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * (int64, int64)"
    assert result.to_list() == [(1, 4), (2, 5), (3, 6)]


def test_complex_inner():
    one = ak.highlevel.Array([1, 2, 3])
    two = ak.highlevel.Array([[7, 5], [1, 2], [4, None]])
    result = ak.operations.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * var * ?(int64, int64)"
    assert result.to_list() == [[(1, 7), (1, 5)], [(2, 1), (2, 2)], [(3, 4), None]]


def test_complex_outer():
    one = ak.highlevel.Array([1, None, 3])
    two = ak.highlevel.Array([[7, 5], [1, 2], [4, None]])
    result = ak.operations.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * option[var * ?(int64, int64)]"
    assert result.to_list() == [[(1, 7), (1, 5)], None, [(3, 4), None]]
