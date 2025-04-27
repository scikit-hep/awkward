# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import ast

import numpy as np  # noqa: F401

import awkward as ak

PREFIX = "foo"


def test_NumpyForm():
    form = ak.forms.NumpyForm("float64")
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)


def test_BitMaskedForm():
    form = ak.forms.BitMaskedForm(
        "i8",
        ak.forms.NumpyForm("float64"),
        True,
        False,
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)
    assert ast.literal_eval(form_with_unique_key.content.form_key) == (PREFIX, None)


def test_EmptyForm():
    form = ak.forms.EmptyForm()
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)


def test_IndexedForm():
    form = ak.forms.IndexedForm(
        "i64",
        ak.forms.NumpyForm("float64"),
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)
    assert ast.literal_eval(form_with_unique_key.content.form_key) == (PREFIX, None)


def test_IndexedOptionForm():
    form = ak.forms.IndexedOptionForm(
        "i64",
        ak.forms.NumpyForm("float64"),
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)
    assert ast.literal_eval(form_with_unique_key.content.form_key) == (PREFIX, None)


def test_ListForm():
    form = ak.forms.ListForm(
        "i64",
        "i64",
        ak.forms.NumpyForm("float64"),
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)
    assert ast.literal_eval(form_with_unique_key.content.form_key) == (PREFIX, None)


def test_ListOffsetForm():
    form = ak.forms.ListOffsetForm(
        "i64",
        ak.forms.NumpyForm("float64"),
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)
    assert ast.literal_eval(form_with_unique_key.content.form_key) == (PREFIX, None)


def test_RecordForm():
    form = ak.forms.RecordForm(
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("bool")],
        ["one", "two"],
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.content("one").form_key) == (
        PREFIX,
        "one",
    )
    assert ast.literal_eval(form_with_unique_key.content("two").form_key) == (
        PREFIX,
        "two",
    )


def test_RegularForm():
    form = ak.forms.RegularForm(
        ak.forms.NumpyForm("float64"),
        10,
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)
    assert ast.literal_eval(form_with_unique_key.content.form_key) == (PREFIX, None)


def test_UnionForm():
    form = ak.forms.UnionForm(
        "i8",
        "i64",
        [ak.forms.NumpyForm("float64"), ak.forms.NumpyForm("bool")],
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.contents[0].form_key) == (PREFIX, 0)
    assert ast.literal_eval(form_with_unique_key.contents[1].form_key) == (PREFIX, 1)


def test_UnmaskedForm():
    form = ak.forms.UnmaskedForm(
        ak.forms.NumpyForm("float64"),
    )
    assert form.form_key is None

    form_with_unique_key = ak.forms.form_with_unique_keys(form, (PREFIX,))
    assert ast.literal_eval(form_with_unique_key.form_key) == (PREFIX,)
    assert ast.literal_eval(form_with_unique_key.content.form_key) == (PREFIX, None)
