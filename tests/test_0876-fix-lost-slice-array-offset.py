# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    i = ak.Array([0, 1, 2, 3, 4, 5])
    x = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    assert x[i[3:4].mask[[True]]].tolist() == [4.4]
    assert x[i[3:4][[True]]].tolist() == [4.4]
