# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    arr = ak.Array([{"x": 1}, {"x": 2}])
    # Fill with
    result = ak.full_like(arr, np.datetime64(20, "s"), dtype="<M8[s]")
    assert result.layout.is_equal_to(
        ak.contents.RecordArray(
            [
                ak.contents.NumpyArray(
                    np.array([20, 20], dtype=np.dtype("datetime64[s]"))
                )
            ],
            ["x"],
        )
    )


def test_typetracer():
    arr = ak.Array([{"x": 1}, {"x": 2}], backend="typetracer")
    # Fill with
    result = ak.full_like(arr, np.datetime64(20, "s"), dtype="<M8[s]")
    assert result.layout.form == (
        ak.forms.RecordForm(
            [ak.forms.NumpyForm("datetime64[s]")],
            ["x"],
        )
    )
