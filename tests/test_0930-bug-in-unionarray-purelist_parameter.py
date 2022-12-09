# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    behavior = {}

    @ak.behaviors.mixins.mixin_class(behavior)
    class Blah:
        @property
        def blah(self):
            return self["x"]

    a = ak.operations.zip({"x": [[1, 2], [3]], "y": [[4, 5], [6]]})
    b = ak.operations.zip({"x": [[-1, -2, -3], [-4]], "z": [[-4, -5, -6], [-7]]})

    a2 = a[a.x % 2 == 0]
    b2 = b[b.x % 2 == 0]
    c2 = ak.operations.with_name(
        ak.operations.concatenate([a2, b2], axis=1), "Blah", behavior=behavior
    )

    assert c2.blah.to_list() == [[2, -2], [-4]]
