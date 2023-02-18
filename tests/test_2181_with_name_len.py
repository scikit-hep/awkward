# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest

import awkward as ak


def test():
    layout = ak.to_layout([{"x": 10}]).to_typetracer(forget_length=True)
    named = ak.with_name(layout, "x_like")
    assert ak.parameters(named) == {"__record__": "x_like"}


@pytest.mark.xfail(reason="TypeTracer.frombuffer not implemented")
def test_strings():
    layout = ak.to_layout(["hello", "world"]).to_typetracer(forget_length=True)
    ak.zeros_like(layout)
