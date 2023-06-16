# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import numpy as np

import awkward as ak


def test():
    layout = ak.contents.ByteMaskedArray(
        ak.index.Index8([0]),
        ak.contents.ListOffsetArray(
            ak.index.Index64([0, 1]),
            ak.contents.ByteMaskedArray(
                ak.index.Index8([1]),
                ak.contents.ListOffsetArray(
                    ak.index.Index64([0, 2]), ak.contents.NumpyArray([5, 6])
                ),
                valid_when=True,
            ),
        ),
        valid_when=True,
    )
    assert ak.almost_equal(
        ak.any(layout, axis=2),
        ak.contents.IndexedOptionArray(
            ak.index.Index64([-1]),
            ak.contents.ListOffsetArray(
                ak.index.Index64([0]),
                ak.contents.IndexedOptionArray(
                    ak.index.Index64([]),
                    ak.contents.NumpyArray(np.empty(0, dtype=np.bool_)),
                ),
            ),
        ),
    )
