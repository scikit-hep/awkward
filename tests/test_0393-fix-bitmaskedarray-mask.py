# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    assert np.asarray(
        ak.layout.ByteMaskedArray(
            ak.layout.Index8(np.array([1, 1, 1, 1, 1], np.int8)),
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4])),
            valid_when=True,
        ).bytemask()
    ).tolist() == [0, 0, 0, 0, 0]

    assert np.asarray(
        ak.layout.ByteMaskedArray(
            ak.layout.Index8(np.array([1, 1, 1, 1, 1], np.int8)),
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4])),
            valid_when=False,
        ).bytemask()
    ).tolist() == [1, 1, 1, 1, 1]

    assert np.asarray(
        ak.layout.BitMaskedArray(
            ak.layout.IndexU8(np.array([31], np.uint8)),
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4])),
            valid_when=True,
            length=5,
            lsb_order=True,
        ).bytemask()
    ).tolist() == [0, 0, 0, 0, 0]

    assert np.asarray(
        ak.layout.BitMaskedArray(
            ak.layout.IndexU8(np.array([31], np.uint8)),
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4])),
            valid_when=False,
            length=5,
            lsb_order=True,
        ).bytemask()
    ).tolist() == [1, 1, 1, 1, 1]
