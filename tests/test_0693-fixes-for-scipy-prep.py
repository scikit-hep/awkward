# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_varnewaxis():
    array = ak.Array([[[ 0,  1,  2,  3,  4],
                       [ 5,  6,  7,  8,  9],
                       [10, 11, 12, 13, 14]],
                      [[15, 16, 17, 18, 19],
                       [20, 21, 22, 23, 24],
                       [25, 26, 27, 28, 29]]])
    slicer = ak.Array([[3, 4],
                       [0, 1, 2, 3]])
    print(array[slicer[:, np.newaxis]].layout)
    raise Exception

# varnewaxis(jagged([0, 2, 6], array([3, 4, 0, 1, 2, 3])))

# jagged([0, 3, 6], jagged([0, 2, 4, 6, 10, 14, 18], array([3, 4, 3, 4, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])))

# [[[ 3,  4],
#   [ 8,  9],
#   [13, 14]],
#  [[15, 16, 17, 18],
#   [20, 21, 22, 23],
#   [25, 26, 27, 28]]]
