# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401

import awkward as ak


def test_from_iter():
    # We define `from_type` to match ArrayBuilder where possible. We can't
    # include options inside unions, though, because ArrayBuilder creates `UnmaskedArray`
    # nodes for the non-indexed option
    array = ak.to_packed(
        ak.from_iter([1, 2, "hi", [3, 4, {"x": 4}], {"y": [None, 2]}])[:0]
    )
    form_from_type = ak.forms.from_type(array.type.content)
    assert form_from_type == array.layout.form


def test_regular():
    array = ak.to_regular([[1, 2, 3]])[:0]
    form_from_type = ak.forms.from_type(array.type.content)
    assert form_from_type == array.layout.form


def test_categorical():
    array = ak.to_categorical(
        [
            1,
            1,
            2,
            1,
            1,
        ]
    )
    form_from_type = ak.forms.from_type(array.type.content)
    assert form_from_type == array.layout.form


def test_categorical_option():
    array = ak.to_categorical([1, 1, 2, 1, 1, None])
    form_from_type = ak.forms.from_type(array.type.content)
    assert form_from_type == array.layout.form
