# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test():
    data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
    assert ak.to_list(ak.sort(data)) == [[5, 7, 7], [], [2], [2, 8]]

    index = ak.argsort(data)
    assert ak.to_list(data[index]) == [[5, 7, 7], [], [2], [2, 8]]
