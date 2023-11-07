# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import awkward as ak


def test():
    layout = ak.to_layout([{"x": 10}]).to_typetracer(forget_length=True)
    named = ak.with_name(layout, "x_like")
    assert ak.parameters(named) == {"__record__": "x_like"}
