from __future__ import annotations

import numpy as np

import awkward as ak
from awkward.typetracer import unknown_length


def test():
    arr = (
        ak.mask([1, 2, 3], [True, False, False], highlevel=False)
        .to_BitMaskedArray(valid_when=True, lsb_order=True)
        .to_typetracer(forget_length=True)
    )

    result = arr.mask_as_bool()
    assert result.dtype == np.dtype("bool")
    assert result.ndim == 1
    assert result.size is unknown_length
