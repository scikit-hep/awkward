# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
import awkward.index
from awkward.contents.bitmaskedarray import BitMaskedArray
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


def test_unique_bitmasked_flat_axis_minus_one():
    """BitMaskedArray._unique(axis=-1) on flat content returns a flat option array.

    For a flat (depth-1) option array, ``axis=-1`` reduces over the single
    top-level group. ``IndexedOptionArray._unique`` wraps that result in a
    ``ListOffsetArray`` (one sublist per group), so ``BitMaskedArray._unique``
    strips the spurious wrapper for ``negaxis is not None`` to yield a flat
    result -- preserving the option type and the trailing ``None``.
    """
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

    bm_result = ak._do.unique(bm_layout, axis=-1)

    # Flat (depth-1) result -- the spurious ListOffsetArray wrapper is stripped,
    # matching the axis=-1 expectation in test_0404_array_validity_check, and the
    # option type is preserved.
    assert bm_result.minmax_depth == (1, 1)
    assert ak.to_list(bm_result) == [1.0, 2.0, 3.0]
    assert "?" in str(bm_result.form.type)
