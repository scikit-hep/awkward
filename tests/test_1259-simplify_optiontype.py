# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test_byte_masked_array():
    layout = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(np.r_[0, 1, 1]),
        ak.contents.IndexedOptionArray(
            ak.index.Index64([0, 1, -1]), ak.contents.NumpyArray(np.arange(3))
        ),
        valid_when=True,
    )
    assert layout.to_list() == [None, 1, None]

    assert isinstance(layout, ak.contents.IndexedOptionArray)
    assert np.asarray(layout.index).tolist() == [-1, 1, -1]


def test_bit_masked_array():
    layout = ak.contents.BitMaskedArray.simplified(
        ak.index.IndexU8(np.array([0 | 1 << 1 | 1 << 2], dtype=np.uint8)),
        ak.contents.IndexedOptionArray(
            ak.index.Index64([0, 1, -1]), ak.contents.NumpyArray(np.arange(3))
        ),
        valid_when=True,
        length=3,
        lsb_order=True,
    )
    assert layout.to_list() == [None, 1, None]

    assert isinstance(layout, ak.contents.IndexedOptionArray)
    assert np.asarray(layout.index).tolist() == [-1, 1, -1]
