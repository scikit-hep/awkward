# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    @ak._v2.behaviors.mixins.mixin_class(ak._v2.behavior)
    class Blah:
        @property
        def blah(self):
            return self["x"]

    a = ak._v2.operations.zip({"x": [[1, 2], [3]], "y": [[4, 5], [6]]})
    b = ak._v2.operations.zip({"x": [[-1, -2, -3], [-4]], "z": [[-4, -5, -6], [-7]]})

    a2 = a[a.x % 2 == 0]
    b2 = b[b.x % 2 == 0]
    c2 = ak._v2.operations.with_name(
        ak._v2.operations.concatenate([a2, b2], axis=1), "Blah"
    )

    assert c2.blah.tolist() == [[2, -2], [-4]]
