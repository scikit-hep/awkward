# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
import awkward.index
from awkward.contents.bitmaskedarray import BitMaskedArray
from awkward.contents.bytemaskedarray import ByteMaskedArray
from awkward.contents.numpyarray import NumpyArray


def test_to_BitMaskedArray_inversion():
    """Round-trip: BitMaskedArray -> to_BitMaskedArray with flipped valid_when."""
    # [None, 1.0, None, 2.0, 3.0, None, None, 4.0]
    # lsb_order=True, valid_when=True: bit=1 => valid
    # bits (LSB first): 0,1,0,1,1,0,0,1 => byte = 0b10011010
    mask_byte = np.array([0b10011010], dtype=np.uint8)
    content = NumpyArray(np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0]))
    bma = BitMaskedArray(
        ak.index.IndexU8(mask_byte),
        content,
        valid_when=True,
        length=8,
        lsb_order=True,
    )
    assert ak.to_list(bma) == [None, 1.0, None, 2.0, 3.0, None, None, 4.0]

    # Flip valid_when: requires bitwise ~ to correctly invert the mask
    inverted = bma.to_BitMaskedArray(valid_when=False, lsb_order=True)
    assert inverted._valid_when is False
    assert ak.to_list(inverted) == [None, 1.0, None, 2.0, 3.0, None, None, 4.0]


def test_unique_bitmasked_consistent_with_bytemasked():
    """BitMaskedArray._unique should not strip content when negaxis is not None."""
    values = [1.0, 1.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0]
    valid = [True, True, True, True, True, False, False, False]

    n = len(values)
    mask_bytes = np.zeros((n + 7) // 8, dtype=np.uint8)
    for i, v in enumerate(valid):
        if v:
            mask_bytes[i // 8] |= 1 << (i % 8)
    content = NumpyArray(np.array(values, dtype=np.float64))

    bm_layout = BitMaskedArray(
        ak.index.IndexU8(mask_bytes), content, valid_when=True, length=n, lsb_order=True
    )

    byte_mask = np.array([1 if v else 0 for v in valid], dtype=np.int8)
    bym_layout = ByteMaskedArray(
        ak.index.Index8(byte_mask),
        NumpyArray(np.array(values, dtype=np.float64)),
        valid_when=True,
    )

    bm_result = ak._do.unique(bm_layout, axis=-1)
    bym_result = ak._do.unique(bym_layout, axis=-1)

    bm_unique = sorted(v for v in ak.to_list(ak.Array(bm_result)) if v is not None)
    bym_unique = sorted(v for v in ak.to_list(ak.Array(bym_result)) if v is not None)

    assert bm_unique == bym_unique
    assert (None in ak.to_list(ak.Array(bm_result))) == (
        None in ak.to_list(ak.Array(bym_result))
    )
