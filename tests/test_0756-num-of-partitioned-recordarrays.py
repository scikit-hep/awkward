# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    array = ak.repartition([{"x": x, "y": x * 10} for x in range(10)], 2)
    assert ak.to_list(ak.num(array.layout, axis=0)) == {"x": 10, "y": 10}
