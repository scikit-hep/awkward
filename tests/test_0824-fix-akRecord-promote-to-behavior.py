# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


class Point(ak.Record):
    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def test():
    behavior = {"point": Point}
    rec = ak.Record({"x": 1, "y": 1.1}, with_name="point", behavior=behavior)
    assert isinstance(rec, Point)
    assert hasattr(rec, "distance")
