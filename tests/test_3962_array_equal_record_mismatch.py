# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_tuple_different_arity():
    a1 = ak.Array([(1,), (2,)])
    a2 = ak.Array([(1, 10), (2, 20)])

    assert ak.array_equal(a1, a2) is False
    assert ak.array_equal(a2, a1) is False


def test_record_different_field_names():
    a1 = ak.Array([{"x": 1}, {"x": 2}])
    a2 = ak.Array([{"y": 1}, {"y": 2}])

    assert ak.array_equal(a1, a2) is False
    assert ak.array_equal(a2, a1) is False


def test_record_field_order_insensitive():
    a1 = ak.Array([{"x": 1, "y": 10}, {"x": 2, "y": 20}])
    a2 = ak.Array([{"y": 10, "x": 1}, {"y": 20, "x": 2}])

    assert ak.array_equal(a1, a2) is True
    assert ak.array_equal(a2, a1) is True
