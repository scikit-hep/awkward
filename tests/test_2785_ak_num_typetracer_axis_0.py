# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak
from awkward._nplikes.typetracer import TypeTracerArray


def test_unknown_length():
    array = ak.typetracer.typetracer_from_form(ak.forms.NumpyForm("int64"))
    assert isinstance(ak.num(array, axis=0), TypeTracerArray)


def test_known_length():
    array = ak.Array([0, 1, 2, 3], backend="typetracer")
    assert ak.num(array, axis=0) == 4