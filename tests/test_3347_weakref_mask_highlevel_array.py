from __future__ import annotations

import pytest

import awkward as ak


def test_Array_mask_weakref():
    arr = ak.Array([1])
    m = arr.mask

    assert ak.all(m[[True]] == arr)

    del arr
    with pytest.raises(
        ValueError,
        match=r"The array to mask was deleted before it could be masked. If you want to construct this mask, you must either keep the array alive or use 'ak.mask' explicitly.",
    ):
        _ = m[[True]]
