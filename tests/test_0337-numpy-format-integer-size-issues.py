# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_index_slice():
    index = awkward1.layout.Index64(numpy.array(
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        numpy.int64))
    assert index[4] == 400
    assert numpy.asarray(index[3:7]).tolist() == [300, 400, 500, 600]

def test_dtype_to_form():
    assert awkward1.forms.Form.from_numpy(numpy.dtype("bool")) == awkward1.forms.NumpyForm((), 1, "?")
    assert awkward1.forms.Form.from_numpy(numpy.dtype("int8")) == awkward1.forms.NumpyForm((), 1, "b")
    assert awkward1.forms.Form.from_numpy(numpy.dtype("int16")) == awkward1.forms.NumpyForm((), 2, "h")
    assert awkward1.forms.Form.from_numpy(numpy.dtype("int32")) == awkward1.forms.Form.fromjson('"int32"')
    assert awkward1.forms.Form.from_numpy(numpy.dtype("int64")) == awkward1.forms.Form.fromjson('"int64"')
    assert awkward1.forms.Form.from_numpy(numpy.dtype("uint8")) == awkward1.forms.NumpyForm((), 1, "B")
    assert awkward1.forms.Form.from_numpy(numpy.dtype("uint16")) == awkward1.forms.NumpyForm((), 2, "H")
    assert awkward1.forms.Form.from_numpy(numpy.dtype("uint32")) == awkward1.forms.Form.fromjson('"uint32"')
    assert awkward1.forms.Form.from_numpy(numpy.dtype("uint64")) == awkward1.forms.Form.fromjson('"uint64"')
    assert awkward1.forms.Form.from_numpy(numpy.dtype("float32")) == awkward1.forms.NumpyForm((), 4, "f")
    assert awkward1.forms.Form.from_numpy(numpy.dtype("float64")) == awkward1.forms.NumpyForm((), 8, "d")
    assert awkward1.forms.Form.from_numpy(numpy.dtype(("float64", (2, 3, 5)))) == awkward1.forms.NumpyForm((2, 3, 5), 8, "d")
