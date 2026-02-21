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
