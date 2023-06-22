# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_dtype_deprecated():
    form = ak.forms.EmptyForm()
    with pytest.warns(
        DeprecationWarning,
        match=r"the `dtype` parameter in EmptyForm\.to_NumpyForm is deprecated",
    ):
        next_form = form.to_NumpyForm(dtype=np.dtype(np.int64))
    assert next_form.primitive == "int64"


def test_primitive():
    form = ak.forms.EmptyForm()
    next_form = form.to_NumpyForm(primitive="int64")
    assert next_form.primitive == "int64"
