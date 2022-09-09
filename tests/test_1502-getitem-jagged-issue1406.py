# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np

to_list = ak._v2.operations.to_list


def test_1406issue():
    array = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([1, 3], dtype=np.int64)),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(np.array([0, 2, 2, 3], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index64(np.array([0, 1], dtype=np.int64)),
                ak._v2.contents.ListOffsetArray(
                    ak._v2.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                    ak._v2.contents.NumpyArray(np.array([0], dtype=np.int64)),
                ),
            ),
        ),
        check_valid=True,
    )
    assert to_list(array[index]) == [[[], [2]]]


def test_success_remove_option_type():
    array = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([1, 3], dtype=np.int64)),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(np.array([0, 2, 2, 3], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([0], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    assert to_list(array[index]) == [[[], [2]]]


def test_success_start_offset0():

    array = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(np.array([2, 2, 3], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index64(np.array([0, 1], dtype=np.int64)),
                ak._v2.contents.ListOffsetArray(
                    ak._v2.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
                    ak._v2.contents.NumpyArray(np.array([0], dtype=np.int64)),
                ),
            ),
        ),
        check_valid=True,
    )

    assert to_list(array[index]) == [[[], [2]]]


def test_success_nonempty_list():
    array = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([1, 3], dtype=np.int64)),
            ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(np.array([0, 1, 2, 3], dtype=np.int64)),
                ak._v2.contents.NumpyArray(np.array([0, 1, 2], dtype=np.int64)),
            ),
        ),
        check_valid=True,
    )

    index = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak._v2.contents.IndexedOptionArray(
                ak._v2.index.Index64(np.array([0, 1], dtype=np.int64)),
                ak._v2.contents.ListOffsetArray(
                    ak._v2.index.Index64(np.array([0, 1, 2], dtype=np.int64)),
                    ak._v2.contents.NumpyArray(np.array([0, 0], dtype=np.int64)),
                ),
            ),
        ),
        check_valid=True,
    )

    assert to_list(array[index]) == [[[1], [2]]]
