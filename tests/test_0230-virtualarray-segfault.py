# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_0230():
    rec = ak.zip(
        {"x": ak.virtual(lambda: ak.Array([1, 2, 3, 4]), length=4)}, depth_limit=1
    )
    assert ak.to_list(rec.x[1:]) == [2, 3, 4]
    assert ak.to_list(rec.x[1:] * 2) == [4, 6, 8]


def test_SliceGenerator():
    layout = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5]))

    generator = ak.layout.SliceGenerator(layout, slice(1, None))

    assert ak.to_list(generator()) == [2, 3, 4, 5]
