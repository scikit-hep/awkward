# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Regression tests for BitMaskedArray mask inversion and placeholder detection fixes (#4092)."""

from __future__ import annotations

import numpy as np

import awkward as ak
import awkward.index
from awkward.contents.bitmaskedarray import BitMaskedArray
from awkward.contents.bytemaskedarray import ByteMaskedArray
from awkward.contents.numpyarray import NumpyArray

# -------------------------------------------------------------------
# Fix 1: to_BitMaskedArray valid_when inversion uses bitwise ~ not logical_not
# -------------------------------------------------------------------


def test_to_BitMaskedArray_inversion_same_lsb_order():
    """Round-trip: BitMaskedArray -> to_BitMaskedArray with flipped valid_when."""
    # Build an array: [None, 1.0, None, 2.0, 3.0, None, None, 4.0]
    # lsb_order=True, valid_when=True means bit=1 => valid
    # Bits for 8 elements (LSB first): 0,1,0,1,1,0,0,1 => byte = 0b10011010 = 0x9A = 154
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

    # Convert to BitMaskedArray with same lsb_order but flipped valid_when
    # Now valid_when=False means bit=0 => valid, bit=1 => None
    inverted = bma.to_BitMaskedArray(valid_when=False, lsb_order=True)
    assert inverted._valid_when is False
    result = ak.to_list(inverted)
    assert result == [None, 1.0, None, 2.0, 3.0, None, None, 4.0], (
        f"Expected [None,1.0,None,2.0,3.0,None,None,4.0], got {result}"
    )


def test_to_BitMaskedArray_inversion_round_trip():
    """Inverting valid_when twice returns the same array."""
    mask_byte = np.array([0b10011010], dtype=np.uint8)
    content = NumpyArray(np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0]))
    bma = BitMaskedArray(
        ak.index.IndexU8(mask_byte),
        content,
        valid_when=True,
        length=8,
        lsb_order=True,
    )
    inverted = bma.to_BitMaskedArray(valid_when=False, lsb_order=True)
    back = inverted.to_BitMaskedArray(valid_when=True, lsb_order=True)
    assert np.array_equal(back._mask.data, bma._mask.data)
    assert ak.to_list(back) == ak.to_list(bma)


def test_to_BitMaskedArray_inversion_big_endian():
    """Same test but with lsb_order=False (big endian bit order)."""
    # lsb_order=False means MSB first: bits index 0..7 are bits 7..0 of the byte
    # For [None, 1.0, None, 2.0, ...]: bits = 0,1,0,1,1,0,0,1 (MSB first)
    # byte = 0b01011001 = 89
    mask_byte = np.array([0b01011001], dtype=np.uint8)
    content = NumpyArray(np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0]))
    bma = BitMaskedArray(
        ak.index.IndexU8(mask_byte),
        content,
        valid_when=True,
        length=8,
        lsb_order=False,
    )
    assert ak.to_list(bma) == [None, 1.0, None, 2.0, 3.0, None, None, 4.0]

    inverted = bma.to_BitMaskedArray(valid_when=False, lsb_order=False)
    result = ak.to_list(inverted)
    assert result == [None, 1.0, None, 2.0, 3.0, None, None, 4.0], (
        f"Expected [None,1.0,None,2.0,3.0,None,None,4.0] got {result}"
    )


# -------------------------------------------------------------------
# Fix 2: _unique returns consistent result for option types
# -------------------------------------------------------------------


def _make_bitmasked(values, valid_mask):
    """Create a BitMaskedArray from a list of values and a boolean valid mask."""
    n = len(values)
    nbytes = (n + 7) // 8
    mask_bytes = np.zeros(nbytes, dtype=np.uint8)
    for i, v in enumerate(valid_mask):
        if v:
            mask_bytes[i // 8] |= 1 << (i % 8)  # LSB order, valid_when=True
    content = NumpyArray(np.array(values, dtype=np.float64))
    return BitMaskedArray(
        ak.index.IndexU8(mask_bytes),
        content,
        valid_when=True,
        length=n,
        lsb_order=True,
    )


def _make_bytemasked(values, valid_mask):
    """Create a ByteMaskedArray from a list of values and a boolean valid mask."""
    mask = np.array([1 if v else 0 for v in valid_mask], dtype=np.int8)
    content = NumpyArray(np.array(values, dtype=np.float64))
    return ByteMaskedArray(
        ak.index.Index8(mask),
        content,
        valid_when=True,
    )


def test_unique_bitmasked_vs_bytemasked_flat():
    """BitMaskedArray and ByteMaskedArray._unique should agree for flat data."""
    values = [1.0, 1.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0]
    valid = [True, True, True, True, True, False, False, False]

    bm_layout = _make_bitmasked(values, valid)
    bym_layout = _make_bytemasked(values, valid)

    bm_result = ak._do.unique(bm_layout, axis=-1)
    bym_result = ak._do.unique(bym_layout, axis=-1)

    bm_unique = sorted(v for v in ak.to_list(ak.Array(bm_result)) if v is not None)
    bym_unique = sorted(v for v in ak.to_list(ak.Array(bym_result)) if v is not None)

    assert bm_unique == bym_unique, (
        f"BitMasked unique={bm_unique}, ByteMasked unique={bym_unique}"
    )
    # Both should include None
    bm_has_none = None in ak.to_list(ak.Array(bm_result))
    bym_has_none = None in ak.to_list(ak.Array(bym_result))
    assert bm_has_none == bym_has_none, (
        f"None presence: BitMasked={bm_has_none}, ByteMasked={bym_has_none}"
    )


def test_unique_bitmasked_axis_minus1():
    """_unique on BitMaskedArray at axis=-1 returns the same structure as ByteMasked."""
    values = [1.0, 1.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0]
    valid = [True, True, True, True, True, False, False, False]
    bm_layout = _make_bitmasked(values, valid)
    result_layout = ak._do.unique(bm_layout, axis=-1)
    result = ak.to_list(ak.Array(result_layout))
    # Should produce unique non-null values (structure mirrors ByteMaskedArray)
    assert isinstance(result, list), f"Expected a list, got {type(result)}"
    # Flatten to get the actual unique values
    flat = [v for sub in result for v in (sub if isinstance(sub, list) else [sub])]
    non_none = sorted(v for v in flat if v is not None)
    assert non_none == [1.0, 2.0, 3.0], (
        f"Expected unique [1.0, 2.0, 3.0], got {non_none}"
    )


def test_unique_bytemasked_axis_minus1():
    """_unique on ByteMaskedArray at axis=-1 returns unique non-null values."""
    values = [1.0, 1.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0]
    valid = [True, True, True, True, True, False, False, False]
    bym_layout = _make_bytemasked(values, valid)
    result_layout = ak._do.unique(bym_layout, axis=-1)
    result = ak.to_list(ak.Array(result_layout))
    assert isinstance(result, list), f"Expected a list, got {type(result)}"
    flat = [v for sub in result for v in (sub if isinstance(sub, list) else [sub])]
    non_none = sorted(v for v in flat if v is not None)
    assert non_none == [1.0, 2.0, 3.0], (
        f"Expected unique [1.0, 2.0, 3.0], got {non_none}"
    )


# -------------------------------------------------------------------
# Fix 3: _is_getitem_at_placeholder checks .data not the Index wrapper
# -------------------------------------------------------------------


def test_is_getitem_at_placeholder_bitmasked_content_check():
    """_is_getitem_at_placeholder propagates to content correctly."""
    mask_byte = np.array([0xFF], dtype=np.uint8)
    content = NumpyArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    bma = BitMaskedArray(
        ak.index.IndexU8(mask_byte),
        content,
        valid_when=True,
        length=8,
        lsb_order=True,
    )
    # Without a PlaceholderArray in the mask data, should return False
    # (content is a NumpyArray which also returns False)
    assert bma._is_getitem_at_placeholder() is False


def test_is_getitem_at_placeholder_bytemasked_content_check():
    """_is_getitem_at_placeholder propagates to content correctly for ByteMasked."""
    mask = np.array([1, 1, 1, 1], dtype=np.int8)
    content = NumpyArray(np.array([1.0, 2.0, 3.0, 4.0]))
    bma = ByteMaskedArray(
        ak.index.Index8(mask),
        content,
        valid_when=True,
    )
    assert bma._is_getitem_at_placeholder() is False
