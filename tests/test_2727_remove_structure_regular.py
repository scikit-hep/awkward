# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.Array([[[1, 2], [1]], [[1], [1], [1]]])
    assert ak.almost_equal(
        ak.mean(array, axis=None, keepdims=True),
        ak.contents.RegularArray(
            ak.contents.RegularArray(
                ak.contents.NumpyArray(
                    np.mean([1, 2, 1, 1, 1, 1], dtype=np.float64, keepdims=True)
                ),
                size=1,
            ),
            size=1,
        ),
        check_regular=True,
    )
