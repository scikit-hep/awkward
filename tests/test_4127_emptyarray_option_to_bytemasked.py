# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_array_equal_empty_option_union():
    layout = ak.contents.UnionArray(
        ak.index.Index8(np.empty(0, dtype=np.int8)),
        ak.index.Index64(np.empty(0, dtype=np.int64)),
        [
            ak.contents.BitMaskedArray(
                ak.index.IndexU8(np.empty(0, dtype=np.uint8)),
                ak.contents.EmptyArray(),
                valid_when=True,
                length=0,
                lsb_order=False,
            ),
            ak.contents.BitMaskedArray(
                ak.index.IndexU8(np.empty(0, dtype=np.uint8)),
                ak.contents.EmptyArray(),
                valid_when=True,
                length=0,
                lsb_order=False,
            ),
        ],
    )
    assert ak.array_equal(layout, layout)


def test_indexedoption_emptyarray_length_zero():
    # An empty option-of-unknown converts fine: the carry is empty, so there are
    # no missing values that need a dummy length-one content to point at.
    layout = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.empty(0, dtype=np.int64)),
        ak.contents.EmptyArray(),
    )
    byte_masked = layout.to_ByteMaskedArray(True)
    assert byte_masked.length == 0
    assert byte_masked.content.is_unknown

    bit_masked = layout.to_BitMaskedArray(True, False)
    assert bit_masked.length == 0
    assert bit_masked.content.is_unknown


def test_indexedoption_emptyarray_length_nonzero_raises():
    # A non-empty all-None option-of-unknown cannot be a ByteMaskedArray, because
    # the content would have to be at least as long as the mask and an EmptyArray
    # is always empty. We expect a clear error rather than an internal one.
    layout = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.full(3, -1, dtype=np.int64)),
        ak.contents.EmptyArray(),
    )
    with pytest.raises(ValueError, match="cannot convert an IndexedOptionArray"):
        layout.to_ByteMaskedArray(True)
