# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    @ak.mixin_class(ak.behavior)
    class Blah(object):
        @property
        def blah(self):
            return self["x"]

    a = ak.zip({"x": [[1, 2], [3]], "y": [[4, 5], [6]]})
    b = ak.zip({"x": [[-1, -2, -3], [-4]], "z": [[-4, -5, -6], [-7]]})

    a2 = a[a.x % 2 == 0]
    b2 = b[b.x % 2 == 0]
    c2 = ak.with_name(ak.concatenate([a2, b2], axis=1), "Blah")

    assert c2.blah.tolist() == [[2, -2], [-4]]
