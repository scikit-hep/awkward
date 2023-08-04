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
