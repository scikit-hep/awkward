# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_false_parameter():
    dtype = np.dtype("datetime64[15us]")
    form = ak.forms.numpyform.from_dtype(dtype)
    assert form.primitive == "datetime64[15us]"
