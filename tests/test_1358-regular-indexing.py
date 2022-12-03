# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.from_numpy(np.arange(4 * 4).reshape(4, 4))
    mask_regular = ak.Array((array > 4).layout.to_RegularArray())
    assert array[mask_regular].tolist() == [
        [],
        [5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]

    mask_numpy = ak.to_numpy(mask_regular)
    assert array[mask_numpy].tolist() == [
        [],
        [5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
