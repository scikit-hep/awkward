# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_byte_masked_array():
    layout = ak._v2.contents.ByteMaskedArray(
        ak._v2.index.Index8(np.r_[0, 1, 1]),
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64([0, 1, -1]), ak._v2.contents.NumpyArray(np.arange(3))
        ),
        valid_when=True,
    )
    assert layout.tolist() == [None, 1, None]

    simplified = layout.simplify_optiontype()
    assert simplified.tolist() == [None, 1, None]

    assert isinstance(simplified, ak._v2.contents.IndexedOptionArray)
    assert np.asarray(simplified.index).tolist() == [-1, 1, -1]


def test_bit_masked_array():
    layout = ak._v2.contents.BitMaskedArray(
        ak._v2.index.IndexU8(np.array([0 | 1 << 1 | 1 << 2], dtype=np.uint8)),
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64([0, 1, -1]), ak._v2.contents.NumpyArray(np.arange(3))
        ),
        valid_when=True,
        length=3,
        lsb_order=True,
    )
    assert layout.tolist() == [None, 1, None]

    simplified = layout.simplify_optiontype()
    assert simplified.tolist() == [None, 1, None]

    assert isinstance(simplified, ak._v2.contents.IndexedOptionArray)
    assert np.asarray(simplified.index).tolist() == [-1, 1, -1]
