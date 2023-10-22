# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak


def test():
    layout = ak.to_layout([{"x": 10}]).to_typetracer(length_policy="drop_recursive")
    named = ak.with_name(layout, "x_like")
    assert ak.parameters(named) == {"__record__": "x_like"}
