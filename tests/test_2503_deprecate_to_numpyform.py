# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test_primitive():
    form = ak.forms.EmptyForm()
    next_form = form.to_NumpyForm(primitive="int64")
    assert next_form.primitive == "int64"
