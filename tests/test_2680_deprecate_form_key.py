# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


def test_unmasked_form():
    with pytest.warns(DeprecationWarning, match=r"form_key"):
        form = ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64"), form_key="llamas")
    assert form.form_key == "llamas"

    form_no_key = ak.forms.UnmaskedForm(ak.forms.NumpyForm("int64"))
    assert form_no_key.form_key is None


def test_record_form():
    with pytest.warns(DeprecationWarning, match=r"form_key"):
        form = ak.forms.RecordForm([], None, form_key="llamas")
    assert form.form_key == "llamas"

    form_no_key = ak.forms.RecordForm([], None)
    assert form_no_key.form_key is None


def test_empty_form():
    with pytest.warns(DeprecationWarning, match=r"form_key"):
        form = ak.forms.EmptyForm(form_key="llamas")
    assert form.form_key == "llamas"

    form_no_key = ak.forms.EmptyForm()
    assert form_no_key.form_key is None
