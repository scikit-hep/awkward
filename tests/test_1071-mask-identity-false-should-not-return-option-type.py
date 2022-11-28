# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1], dtype=np.int64)),
        ak.contents.IndexedArray(
            ak.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64)),
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4])), 4
            ),
        ),
    )
    array = ak.highlevel.Array(layout)

    assert (
        str(ak.operations.min(array, axis=-1, mask_identity=False).type)
        == "1 * var * float64"
    )
