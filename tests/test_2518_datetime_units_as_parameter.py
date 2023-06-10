# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_no_parameter():
    dtype = np.dtype("datetime64[15us]")
    with pytest.warns(
        DeprecationWarning, match=r"from_dtype conversion of temporal units"
    ):
        form = ak.forms.numpyform.from_dtype(dtype)
    assert form.primitive == "datetime64"
    assert form.parameters == {"__unit__": "15us"}


def test_true_parameter():
    dtype = np.dtype("datetime64[15us]")
    with pytest.warns(
        DeprecationWarning, match=r"from_dtype conversion of temporal units"
    ):
        form = ak.forms.numpyform.from_dtype(dtype, time_units_as_parameter=True)
    assert form.primitive == "datetime64"
    assert form.parameters == {"__unit__": "15us"}


def test_false_parameter():
    dtype = np.dtype("datetime64[15us]")
    form = ak.forms.numpyform.from_dtype(dtype, time_units_as_parameter=False)
    assert form.primitive == "datetime64[15us]"
