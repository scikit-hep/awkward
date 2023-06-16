# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak


def test():
    array = ak.Array([1, [2, 3]])
    result = array + array

    array_tt = ak.Array(array.layout.to_typetracer(forget_length=True))
    result_tt = array_tt + array_tt

    assert result_tt.layout.form == result.layout.form
