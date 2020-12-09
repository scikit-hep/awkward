# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert ak.flatten(ak.Array([[1, 2, 3], [], [4, 5]]), axis=0).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]
    assert ak.flatten(ak.Array([1, 2, 3, 4, 5]), axis=0).tolist() == [1, 2, 3, 4, 5]
    assert ak.flatten(ak.Array([[1, 2, 3], [], [4, 5]]), axis=-2).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]
    assert ak.flatten(ak.Array([1, 2, 3, 4, 5]), axis=-1).tolist() == [1, 2, 3, 4, 5]
