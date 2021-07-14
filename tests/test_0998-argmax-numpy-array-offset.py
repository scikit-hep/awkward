# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

array = ak.Array(
    ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.r_[1, 2, 4, 7]),
        ak.layout.NumpyArray(
            np.r_[
                1.8125,
                0.8125,
                -0.9375,
                1.1875,
                -0.6875,
                1.3125,
                21.3125,
            ]
        ),
    )
)


def test_argmax():
    i = ak.argmax(array, axis=-1)
    assert ak.to_list(i.layout.content) == [0, 1, 2]


def test_argmin():
    i = ak.argmin(array, axis=-1)
    assert ak.to_list(i.layout.content) == [0, 0, 0]
