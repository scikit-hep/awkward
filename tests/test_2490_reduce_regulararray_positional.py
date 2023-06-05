# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    # This example is designed to test a few assumptions about reduction in the regular reduction pathway
    # The outer list introduces _unique parents_ for local (axis>0) reduction
    # The child option introduces a missing parent: parents become _non-dense_
    # The regular array tests our regular reduction pathway
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64([0, 3, 4, 6]),
        ak.contents.ByteMaskedArray(
            ak.index.Index8([1, 1, 1, 0, 1, 1]),
            ak.contents.RegularArray(
                ak.contents.NumpyArray(
                    np.array(
                        [0, 2, 1, 3, 1, 0, 1, 0, 3, 9, 1, 7, 7, 6, 5, 7, 4, 8],
                        dtype=np.int64,
                    )
                ),
                size=3,
            ),
            valid_when=True,
        ),
    )
    result = ak.argmax(layout, axis=1, highlevel=False, mask_identity=False)
    assert result.is_equal_to(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(
                np.array([1, 0, 2, -1, -1, -1, 0, 0, 1], dtype=np.int64)
            ),
            size=3,
        )
    )
