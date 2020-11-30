# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_index_slice():
    index = ak.layout.Index64(
        np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900], np.int64)
    )
    assert index[4] == 400
    assert np.asarray(index[3:7]).tolist() == [300, 400, 500, 600]


def test_dtype_to_form():
    assert ak.forms.Form.from_numpy(np.dtype("bool")) == ak.forms.NumpyForm((), 1, "?")
    assert ak.forms.Form.from_numpy(np.dtype("int8")) == ak.forms.NumpyForm((), 1, "b")
    assert ak.forms.Form.from_numpy(np.dtype("int16")) == ak.forms.NumpyForm((), 2, "h")
    assert ak.forms.Form.from_numpy(np.dtype("int32")) == ak.forms.Form.fromjson(
        '"int32"'
    )
    assert ak.forms.Form.from_numpy(np.dtype("int64")) == ak.forms.Form.fromjson(
        '"int64"'
    )
    assert ak.forms.Form.from_numpy(np.dtype("uint8")) == ak.forms.NumpyForm((), 1, "B")
    assert ak.forms.Form.from_numpy(np.dtype("uint16")) == ak.forms.NumpyForm(
        (), 2, "H"
    )
    assert ak.forms.Form.from_numpy(np.dtype("uint32")) == ak.forms.Form.fromjson(
        '"uint32"'
    )
    assert ak.forms.Form.from_numpy(np.dtype("uint64")) == ak.forms.Form.fromjson(
        '"uint64"'
    )
    assert ak.forms.Form.from_numpy(np.dtype("float32")) == ak.forms.NumpyForm(
        (), 4, "f"
    )
    assert ak.forms.Form.from_numpy(np.dtype("float64")) == ak.forms.NumpyForm(
        (), 8, "d"
    )
    assert ak.forms.Form.from_numpy(
        np.dtype(("float64", (2, 3, 5)))
    ) == ak.forms.NumpyForm((2, 3, 5), 8, "d")
