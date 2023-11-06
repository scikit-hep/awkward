# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

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
    pytest.importorskip("pyarrow")

    array = ak.str.to_categorical(["do", "re", "mi", "fa", "so"])
    form_from_type = ak.forms.from_type(array.type.content)
    assert form_from_type == array.layout.form


def test_categorical_option():
    pytest.importorskip("pyarrow")

    array = ak.str.to_categorical(["do", "re", "mi", "fa", "so", None])
    form_from_type = ak.forms.from_type(array.type.content)
    assert form_from_type == array.layout.form
