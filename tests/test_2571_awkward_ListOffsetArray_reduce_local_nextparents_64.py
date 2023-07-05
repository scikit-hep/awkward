# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize(
    "indextype", [ak.index.Index32, ak.index.IndexU32, ak.index.Index64]
)
def test(indextype):
    array = ak.Array(
        ak.contents.ListOffsetArray(
            indextype(np.array([0, 3, 3, 5, 6, 9])),
            ak.contents.NumpyArray(np.array([6, 9, 9, 4, 4, 2, 5, 2, 7], np.int64)),
        )
    )
    if indextype is ak.index.Index32:
        assert array.layout.offsets.data.dtype == np.dtype(np.int32)
    elif indextype is ak.index.IndexU32:
        assert array.layout.offsets.data.dtype == np.dtype(np.uint32)
    elif indextype is ak.index.Index64:
        assert array.layout.offsets.data.dtype == np.dtype(np.int64)

    assert ak.argsort(array).tolist() == [[0, 1, 2], [], [0, 1], [0], [1, 0, 2]]
