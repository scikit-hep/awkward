# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("pyarrow")


def test_uniques():
    assert ak.str.uniques(["foo", "bar", "bar", "fee", None, "foo"]).tolist() == [
        "foo",
        "bar",
        "fee",
    ]
    assert ak.str.uniques([b"foo", b"bar", b"bar", b"fee", None, b"foo"]).tolist() == [
        b"foo",
        b"bar",
        b"fee",
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
    ]
    assert ak.str.distinct_counts(
        [b"foo", b"bar", b"bar", b"fee", None, b"foo"]
    ).tolist() == [
        {"values": b"foo", "counts": 2},
        {"values": b"bar", "counts": 2},
        {"values": b"fee", "counts": 1},
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


def test_nulls_skipped_at_every_depth():
    assert ak.str.uniques(["a", None, "a"]).tolist() == ["a"]
    assert ak.str.uniques([["a", None, "a"]]).tolist() == [["a"]]
    assert ak.str.distinct_counts(["a", None, "a"]).tolist() == [
        {"values": "a", "counts": 2}
    ]
    assert ak.str.distinct_counts([["a", None, "a"]]).tolist() == [
        [{"values": "a", "counts": 2}]
    ]


@pytest.mark.parametrize("func", [ak.str.uniques, ak.str.distinct_counts])
@pytest.mark.parametrize(
    "array",
    [
        ["a", None, "a"],
        ak.Array(["a", None, None])[1:],
        [["a", None, "a"], None, []],
    ],
)
def test_optional_typetracer_forms(func, array):
    array = ak.Array(array)

    assert (
        func(array).layout.form == func(ak.to_backend(array, "typetracer")).layout.form
    )


def test_regular_lists():
    array = ak.to_regular(ak.Array([["a", "a"], ["b", "c"]]))

    assert ak.str.uniques(array).tolist() == [["a"], ["b", "c"]]
    assert ak.str.distinct_counts(array).tolist() == [
        [{"values": "a", "counts": 2}],
        [{"values": "b", "counts": 1}, {"values": "c", "counts": 1}],
    ]

    typetracer = ak.to_backend(array, "typetracer")
    assert ak.str.uniques(array).layout.form == ak.str.uniques(typetracer).layout.form
    assert (
        ak.str.distinct_counts(array).layout.form
        == ak.str.distinct_counts(typetracer).layout.form
    )


@pytest.mark.parametrize("func", [ak.str.uniques, ak.str.distinct_counts])
@pytest.mark.parametrize(
    "array",
    [
        ["a", 1, "a", 2, "b"],
        [{"x": "a", "y": "c"}, {"x": "a", "y": "c"}],
        [[{"x": "a"}, {"x": "a"}]],
        [["a", 1, "a"]],
    ],
)
def test_strings_in_records_or_unions_raise(func, array):
    with pytest.raises(TypeError, match="record or union"):
        func(array)


def test_lists_of_strings_in_records_and_unions():
    record = ak.Array([{"x": ["a", "a", "b"], "y": 1}, {"x": [], "y": 2}])
    out = ak.str.uniques(record)
    assert ak.validity_error(out) == ""
    assert out.tolist() == [{"x": ["a", "b"], "y": 1}, {"x": [], "y": 2}]

    union = ak.Array([["a", "a"], 1, ["b", None, "b"]])
    out = ak.str.distinct_counts(union)
    assert ak.validity_error(out) == ""
    assert out.tolist() == [
        [{"values": "a", "counts": 2}],
        1,
        [{"values": "b", "counts": 2}],
    ]


@pytest.mark.parametrize("func", [ak.str.uniques, ak.str.distinct_counts])
def test_record_raises(func):
    with pytest.raises(TypeError, match="Record objects are not allowed"):
        func(ak.Record({"x": ["a", "a"]}))


@pytest.mark.parametrize("func", [ak.str.uniques, ak.str.distinct_counts])
@pytest.mark.parametrize(
    "array",
    [
        ak.str.to_categorical(["a", "b", "a", None]),
        ak.str.to_categorical([["a", "b", "a"], ["b", None]]),
    ],
)
def test_categorical(func, array):
    plain = ak.Array(array.to_list())

    assert func(array).tolist() == func(plain).tolist()
    assert func(array).layout.form == func(plain).layout.form
    assert (
        func(array).layout.form == func(ak.to_backend(array, "typetracer")).layout.form
    )


def test_parameters_are_kept():
    strings = ak.with_parameter(ak.Array(["a", "a", "b"]), "foo", "bar")
    array = ak.with_parameter(ak.unflatten(strings, [3]), "__list__", "Foo")

    out = ak.str.uniques(array)
    assert out.layout.parameter("__list__") == "Foo"
    assert out.layout.content.parameter("foo") == "bar"
    assert ak.str.uniques(strings).layout.parameter("foo") == "bar"

    out = ak.str.distinct_counts(array)
    assert out.layout.parameter("__list__") == "Foo"
    assert out.layout.content.content("values").parameter("foo") == "bar"
    out = ak.str.distinct_counts(strings)
    assert out.layout.content("values").parameter("foo") == "bar"


@pytest.mark.parametrize("func", [ak.str.uniques, ak.str.distinct_counts])
def test_typetracer_keeps_list_length(func):
    array = ak.to_backend(ak.Array([["a", "a"], ["b"], []]), "typetracer")

    assert func(array).layout.length == 3


@pytest.mark.parametrize("func", [ak.str.uniques, ak.str.distinct_counts])
def test_no_strings_unchanged(func):
    array = ak.Array(np.arange(4).reshape(2, 2))

    assert func(array).layout.form == array.layout.form
    assert func(array).tolist() == [[0, 1], [2, 3]]


@pytest.mark.parametrize("name", ["uniques", "distinct_counts"])
def test_experimental_warning(name):
    # the warning is issued once per process so check it in a fresh one
    code = rf"""
import re
import warnings

import awkward as ak

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ak.str.{name}(["a", "b", "a"])

assert any(
    issubclass(w.category, ak.errors.ExperimentalWarning)
    and re.match(r"{name} is experimental\.", str(w.message))
    for w in caught
)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
