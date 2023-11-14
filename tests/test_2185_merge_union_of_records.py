# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np  # noqa: F401

import awkward as ak


def test_merge_union_of_records():
    a1 = ak.Array([{"a": 1, "b": 2}])
    a2 = ak.Array([{"b": 3.3, "c": 4.4}])
    c = ak.concatenate((a1, a2))

    assert c.tolist() == [{"a": 1, "b": 2}, {"b": 3.3, "c": 4.4}]

    assert str(c.type) == "2 * union[{a: int64, b: int64}, {b: float64, c: float64}]"

    d = ak.merge_union_of_records(c)

    assert d.tolist() == [{"a": 1, "b": 2, "c": None}, {"a": None, "b": 3.3, "c": 4.4}]

    assert str(d.type) == "2 * {a: ?int64, b: float64, c: ?float64}"


def test_merge_union_of_records_2():
    a1 = ak.Array([{"a": 1, "b": 2}])
    a2 = ak.Array([{"b": 3.3, "c": 4.4}, {"b": None, "c": None}])
    c = ak.concatenate((a1, a2))

    assert c.tolist() == [
        {"a": 1, "b": 2},
        {"b": 3.3, "c": 4.4},
        {"b": None, "c": None},
    ]

    assert str(c.type) == "3 * union[{a: int64, b: int64}, {b: ?float64, c: ?float64}]"

    d = ak.merge_union_of_records(c)

    assert d.tolist() == [
        {"a": 1, "b": 2, "c": None},
        {"a": None, "b": 3.3, "c": 4.4},
        {"a": None, "b": None, "c": None},
    ]

    assert str(d.type) == "3 * {a: ?int64, b: ?float64, c: ?float64}"


def test_merge_union_of_records_3():
    a1 = ak.Array([[[[{"a": 1, "b": 2}]]]])
    a2 = ak.Array([[[[{"b": 3.3, "c": 4.4}]]]])
    c = ak.concatenate((a1, a2), axis=-1)

    assert c.tolist() == [[[[{"a": 1, "b": 2}, {"b": 3.3, "c": 4.4}]]]]

    assert (
        str(c.type)
        == "1 * var * var * var * union[{a: int64, b: int64}, {b: float64, c: float64}]"
    )

    d = ak.merge_union_of_records(c, axis=-1)

    assert d.tolist() == [
        [[[{"a": 1, "b": 2, "c": None}, {"a": None, "b": 3.3, "c": 4.4}]]]
    ]

    assert str(d.type) == "1 * var * var * var * {a: ?int64, b: float64, c: ?float64}"


def test_merge_option_of_records():
    a = ak.Array([None, {"a": 1, "b": 2}])

    assert str(a.type) == "2 * ?{a: int64, b: int64}"

    b = ak.merge_option_of_records(a)

    assert b.tolist() == [{"a": None, "b": None}, {"a": 1, "b": 2}]

    assert str(b.type) == "2 * {a: ?int64, b: ?int64}"


def test_merge_option_of_records_2():
    a = ak.Array([None, {"a": 1, "b": 2}, {"a": None, "b": None}])

    assert str(a.type) == "3 * ?{a: ?int64, b: ?int64}"

    b = ak.merge_option_of_records(a)

    assert b.tolist() == [
        {"a": None, "b": None},
        {"a": 1, "b": 2},
        {"a": None, "b": None},
    ]

    assert str(b.type) == "3 * {a: ?int64, b: ?int64}"


def test_merge_option_of_records_3():
    a = ak.Array([[[[None, {"a": 1, "b": 2}]]]])

    assert str(a.type) == "1 * var * var * var * ?{a: int64, b: int64}"

    b = ak.merge_option_of_records(a, axis=-1)

    assert b.tolist() == [[[[{"a": None, "b": None}, {"a": 1, "b": 2}]]]]

    assert str(b.type) == "1 * var * var * var * {a: ?int64, b: ?int64}"
