# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test():
    a, b, c = ak.broadcast_fields(
        [{"x": 1, "c": 4}],
        [{"y": 2, "c": 5}],
        [{"z": 3, "c": 6}],
    )
    assert ak.almost_equal(a, [{"x": 1, "c": 4, "y": None, "z": None}])
    assert ak.almost_equal(b, [{"x": None, "c": 5, "y": 2, "z": None}])
    assert ak.almost_equal(c, [{"x": None, "c": 6, "y": None, "z": 3}])


def test_nested():
    a, b, c = ak.broadcast_fields(
        [{"x": 1, "z": {"a": 2, "b": 3}}],
        [{"y": 2}],
        [{"y": 3, "z": {"c": 9}}],
    )
    assert ak.almost_equal(a, [{"x": 1, "z": {"a": 2, "b": 3, "c": None}, "y": None}])
    assert ak.almost_equal(b, [{"x": None, "z": None, "y": 2}])
    assert ak.almost_equal(
        c, [{"x": None, "z": {"a": None, "b": None, "c": 9}, "y": 3}]
    )
