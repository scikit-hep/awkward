# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_UnmaskedArray():
    content_float64 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.float64))
    array_float64 = awkward1.layout.UnmaskedArray(content_float64)
    assert awkward1.to_list(array_float64) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert str(awkward1.type(content_float64)) == "float64"
    assert str(awkward1.type(awkward1.Array(content_float64))) == "5 * float64"
    assert str(awkward1.type(array_float64)) == "?float64"
    assert str(awkward1.type(awkward1.Array(array_float64))) == "5 * ?float64"

    assert awkward1.can_cast(numpy.float32, numpy.float64) == True
    assert awkward1.can_cast(numpy.float64, numpy.float32, 'unsafe') == True
    assert awkward1.can_cast(numpy.float64, numpy.int8, 'unsafe') == True

    # content_float32 = awkward1.astype(content_float64, dtype=numpy.float32)
    # array_float32 = awkward1.layout.UnmaskedArray(content_float32)
    # assert awkward1.to_list(array_float32) == [1.1, 2.2, 3.3, 4.4, 5.5]
    # assert str(awkward1.type(content_float32)) == "float32"
    # assert str(awkward1.type(awkward1.Array(content_float32))) == "5 * float32"
    # assert str(awkward1.type(array_float32)) == "?float32"
    # assert str(awkward1.type(awkward1.Array(array_float32))) == "5 * ?float32"

    # content_int8 = awkward1.astype(content_float32, dtype=numpy.int8)
    # array_int8 = awkward1.layout.UnmaskedArray(content_int8)
    # assert awkward1.to_list(array_int8) == [1, 2, 3, 4, 6]
    # assert str(awkward1.type(content_int8)) == "int8"
    # assert str(awkward1.type(awkward1.Array(content_int8))) == "5 * int8"
    # assert str(awkward1.type(array_int8)) == "?int8"
    # assert str(awkward1.type(awkward1.Array(array_int8))) == "5 * ?int8"

# def test_dtype_to_form():
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("bool")) == awkward1.forms.NumpyForm((), 1, "?")
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("int8")) == awkward1.forms.NumpyForm((), 1, "b")
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("int16")) == awkward1.forms.NumpyForm((), 2, "h")
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("int32")) == awkward1.forms.Form.fromjson('"int32"')
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("int64")) == awkward1.forms.Form.fromjson('"int64"')
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("uint8")) == awkward1.forms.NumpyForm((), 1, "B")
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("uint16")) == awkward1.forms.NumpyForm((), 2, "H")
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("uint32")) == awkward1.forms.Form.fromjson('"uint32"')
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("uint64")) == awkward1.forms.Form.fromjson('"uint64"')
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("float32")) == awkward1.forms.NumpyForm((), 4, "f")
#     assert awkward1.forms.Form.from_numpy(numpy.dtype("float64")) == awkward1.forms.NumpyForm((), 8, "d")
#     assert awkward1.forms.Form.from_numpy(numpy.dtype(("float64", (2, 3, 5)))) == awkward1.forms.NumpyForm((2, 3, 5), 8, "d")
