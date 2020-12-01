# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    data = ak.Array([[1, 3, 5, 4, 2], [], [2, 3, 1], [5]])
    assert ak.min(data, axis=1, initial=4).tolist() == [1, None, 1, 4]
    assert ak.max(data, axis=1, initial=4).tolist() == [5, None, 4, 5]

    data = ak.Array([[1.1, 3.3, 5.5, 4.4, 2.2], [], [2.2, 3.3, 1.1], [5.5]])
    assert ak.min(data, axis=1, initial=4).tolist() == [1.1, None, 1.1, 4]
    assert ak.max(data, axis=1, initial=4).tolist() == [5.5, None, 4, 5.5]
