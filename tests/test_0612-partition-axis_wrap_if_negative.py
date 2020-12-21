# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert ak.flatten(
        ak.repartition(ak.Array([[0, 1, 2], [], [3, 4], [5], [6, 7, 8, 9]]), 3)
    ).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
