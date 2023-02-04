# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    assert ak.almost_equal(
        ak.concatenate([ak.Array([1, 2, 3]), ak.Array([1, 2, None])]),
        ak.contents.ByteMaskedArray(
            ak.index.Index8(np.array([False, False, False, False, False, True])),
            ak.contents.NumpyArray(np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)),
            valid_when=False,
        ),
    )
