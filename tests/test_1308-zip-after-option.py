# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401


def test_all_options():
    one = ak.Array([1, 2, None])
    two = ak.Array([None, 5, None])
    result = ak.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * ?(int64, int64)"
    assert result.tolist() == [None, (2, 5), None]


def test_mixed_options():
    one = ak.Array([1, 2, None])
    two = ak.Array([4, 5, 6])
    result = ak.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * ?(int64, int64)"
    assert result.tolist() == [(1, 4), (2, 5), None]


def test_no_options():
    one = ak.Array([1, 2, 3])
    two = ak.Array([4, 5, 6])
    result = ak.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * (int64, int64)"
    assert result.tolist() == [(1, 4), (2, 5), (3, 6)]


def test_complex_inner():
    one = ak.Array([1, 2, 3])
    two = ak.Array([[7, 5], [1, 2], [4, None]])
    result = ak.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * var * ?(int64, int64)"
    assert result.tolist() == [[(1, 7), (1, 5)], [(2, 1), (2, 2)], [(3, 4), None]]


def test_complex_outer():
    one = ak.Array([1, None, 3])
    two = ak.Array([[7, 5], [1, 2], [4, None]])
    result = ak.zip([one, two], optiontype_outside_record=True)
    assert str(result.type) == "3 * option[var * ?(int64, int64)]"
    assert result.tolist() == [[(1, 7), (1, 5)], None, [(3, 4), None]]
