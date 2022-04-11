# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np


def test_list_option_list():
    content = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([1, 3], dtype=np.int64)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2, 2, 3], dtype=np.int64)),
            ak._v2.contents.NumpyArray(np.array([2, 2, 2], dtype=np.int64)),
        ),
    )

    index = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index64(np.array([0, 1], dtype=np.int64)),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ),
        ),
    )

    assert content[index].to_list() == [[[], [2]]]


def test_list_list():
    content = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 0, 1, 3], dtype=np.int64)),
            ak._v2.contents.NumpyArray(np.array([2, 2, 2], dtype=np.int64)),
        ),
    )

    index = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
            ak._v2.contents.NumpyArray(np.array([0], dtype=np.int64)),
        ),
    )

    assert content[index].to_list() == [[[], [2]]]
