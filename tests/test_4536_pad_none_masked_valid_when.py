# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

CONTENT = ak.contents.ListOffsetArray(
    ak.index.Index64(np.array([0, 1, 1, 4])),
    ak.contents.NumpyArray(np.array([1, 2, 3, 4])),
)  # [[1], [], [2, 3, 4]]; the middle list is masked below


def _byte_masked(valid_when):
    mask = np.array([1, 0, 1] if valid_when else [0, 1, 0], dtype=np.int8)
    return ak.contents.ByteMaskedArray(
        ak.index.Index8(mask), CONTENT, valid_when=valid_when
    )


def _bit_masked(valid_when):
    mask = np.array([0b101 if valid_when else 0b010], dtype=np.uint8)
    return ak.contents.BitMaskedArray(
        ak.index.IndexU8(mask), CONTENT, valid_when=valid_when, length=3, lsb_order=True
    )


@pytest.mark.parametrize("make", [_byte_masked, _bit_masked])
@pytest.mark.parametrize("valid_when", [True, False])
@pytest.mark.parametrize("clip", [False, True])
def test_pad_none_axis1(make, valid_when, clip):
    array = ak.Array(make(valid_when))
    assert array.to_list() == [[1], None, [2, 3, 4]]
    expected = [[1, None], None, [2, 3] if clip else [2, 3, 4]]
    assert ak.pad_none(array, 2, axis=1, clip=clip).to_list() == expected
