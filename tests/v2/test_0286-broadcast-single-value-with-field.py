# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_broadcast_single_bool():
    base = ak._v2.Array(
        [[{"x": 0.1, "y": 0.2, "z": 0.3}, {"x": 0.4, "y": 0.5, "z": 0.6}]]
    )
    base_new1 = ak._v2.operations.with_field(base, True, "always_true")
    assert to_list(base_new1.always_true) == [[True, True]]
    base_new2 = ak._v2.operations.with_field(base_new1, base.x > 0.3, "sometimes_true")
    assert to_list(base_new2.always_true) == [[True, True]]
    assert ak._v2.operations.fields(base_new2) == [
        "x",
        "y",
        "z",
        "always_true",
        "sometimes_true",
    ]
