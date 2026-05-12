# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

pytest.importorskip("pyarrow")


def test_uniques():
    assert ak.str.uniques(["foo", "bar", "bar", "fee", None, "foo"]).tolist() == [
        "foo",
        "bar",
        "fee",
        None,
    ]
    assert ak.str.uniques([b"foo", b"bar", b"bar", b"fee", None, b"foo"]).tolist() == [
        b"foo",
        b"bar",
        b"fee",
        None,
    ]

    assert (
        ak.str.uniques(["foo", "bar", "bar", "fee", "foo"]).layout.form
        == ak.str.uniques(
            ak.to_backend(["foo", "bar", "bar", "fee", "foo"], "typetracer")
        ).layout.form
    )
    assert (
        ak.str.uniques([b"foo", b"bar", b"bar", b"fee", b"foo"]).layout.form
        == ak.str.uniques(
            ak.to_backend([b"foo", b"bar", b"bar", b"fee", b"foo"], "typetracer")
        ).layout.form
    )


def test_distinct_counts():
    assert ak.str.distinct_counts(
        ["foo", "bar", "bar", "fee", None, "foo"]
    ).tolist() == [
        {"values": "foo", "counts": 2},
        {"values": "bar", "counts": 2},
        {"values": "fee", "counts": 1},
        {"values": None, "counts": 1},
    ]
    assert ak.str.distinct_counts(
        [b"foo", b"bar", b"bar", b"fee", None, b"foo"]
    ).tolist() == [
        {"values": b"foo", "counts": 2},
        {"values": b"bar", "counts": 2},
        {"values": b"fee", "counts": 1},
        {"values": None, "counts": 1},
    ]

    assert (
        ak.str.distinct_counts(["foo", "bar", "bar", "fee", "foo"]).layout.form
        == ak.str.distinct_counts(
            ak.to_backend(["foo", "bar", "bar", "fee", "foo"], "typetracer")
        ).layout.form
    )
    assert (
        ak.str.distinct_counts([b"foo", b"bar", b"bar", b"fee", b"foo"]).layout.form
        == ak.str.distinct_counts(
            ak.to_backend([b"foo", b"bar", b"bar", b"fee", b"foo"], "typetracer")
        ).layout.form
    )


def test_uniques_nested():
    assert ak.str.uniques([["a", "b", "a"], ["b", "c", "b"]]).tolist() == [
        ["a", "b"],
        ["b", "c"],
    ]


def test_distinct_counts_nested():
    assert ak.str.distinct_counts([["a", "b", "a"], ["b", "c", "b"]]).tolist() == [
        [
            {"values": "a", "counts": 2},
            {"values": "b", "counts": 1},
        ],
        [
            {"values": "b", "counts": 2},
            {"values": "c", "counts": 1},
        ],
    ]


def test_nested_options_empty_and_nulls():
    array = [[[None, None], ["a", None, "a"], []], None, [["b", "b"]]]

    assert ak.str.uniques(array).tolist() == [
        [[], ["a"], []],
        None,
        [["b"]],
    ]
    assert ak.str.distinct_counts(array).tolist() == [
        [
            [],
            [{"values": "a", "counts": 2}],
            [],
        ],
        None,
        [[{"values": "b", "counts": 2}]],
    ]


def test_nested_typetracer_forms():
    array = ak.Array([[["a", "b", "a"], ["b", "c", "b"]], [["x", "x"]]])

    assert (
        ak.str.uniques(array).layout.form
        == ak.str.uniques(ak.to_backend(array, "typetracer")).layout.form
    )
    assert (
        ak.str.distinct_counts(array).layout.form
        == ak.str.distinct_counts(ak.to_backend(array, "typetracer")).layout.form
    )


def test_nested_attrs():
    array = ak.Array([[["a", "b", "a"]]], attrs={"note": "keep"})

    assert ak.str.uniques(array).attrs == {"note": "keep"}
    assert ak.str.distinct_counts(array).attrs == {"note": "keep"}
