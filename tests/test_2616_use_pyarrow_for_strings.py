# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak

pytest.importorskip("pyarrow")

string = ak.Array(
    [
        ["\u03b1\u03b2\u03b3", ""],
        [],
        ["\u2192\u03b4\u03b5\u2190", "\u03b6z z\u03b6", "abc"],
    ]
)
bytestring = ak.Array(
    [
        ["\u03b1\u03b2\u03b3".encode(), b""],
        [],
        ["\u2192\u03b4\u03b5\u2190".encode(), "\u03b6z z\u03b6".encode(), b"abc"],
    ]
)


def test_is_alnum():
    assert ak.str.is_alnum(string).tolist() == [
        [True, False],
        [],
        [False, False, True],
    ]
    assert ak.str.is_alnum(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, True],
    ]


def test_is_alpha():
    assert ak.str.is_alpha(string).tolist() == [
        [True, False],
        [],
        [False, False, True],
    ]
    assert ak.str.is_alpha(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, True],
    ]


def test_is_decimal():
    assert ak.str.is_decimal(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_decimal(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_digit():
    assert ak.str.is_digit(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_digit(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_lower():
    assert ak.str.is_lower(string).tolist() == [
        [True, False],
        [],
        [True, True, True],
    ]
    assert ak.str.is_lower(bytestring).tolist() == [
        [False, False],
        [],
        [False, True, True],
    ]


def test_is_numeric():
    assert ak.str.is_numeric(string).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]
    assert ak.str.is_numeric(bytestring).tolist() == [
        [False, False],
        [],
        [False, False, False],
    ]


def test_is_printable():
    assert ak.str.is_printable(string).tolist() == [
        [True, True],
        [],
        [True, True, True],
    ]
    assert ak.str.is_printable(bytestring).tolist() == [
        [False, True],
        [],
        [False, False, True],
    ]
