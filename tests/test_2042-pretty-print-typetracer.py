# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def test():
    form = ak.forms.ListOffsetForm(
        "i64", ak.forms.NumpyForm("float32", inner_shape=(2,))
    )
    layout = form.length_zero_array().layout.to_typetracer(forget_length=True)
    array = ak.Array(layout)
    assert str(array) == "<Array-typetracer type='?? * var * 2 * float32'>"
