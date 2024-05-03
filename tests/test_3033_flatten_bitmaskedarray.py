from __future__ import annotations

import numpy as np

import awkward as ak


def test():
    layout = ak.to_layout([[[1, 2, 3], [4]], None]).to_BitMaskedArray(
        valid_when=True, lsb_order=True
    )
    result = ak.flatten(layout, axis=1, highlevel=False)
    assert result.is_equal_to(
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 3, 4]),
            ak.contents.NumpyArray(np.array([1, 2, 3, 4], dtype=np.int64)),
        )
    )
