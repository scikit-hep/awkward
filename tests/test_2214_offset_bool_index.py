# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    # fmt: off
    layout = ak.contents.RecordArray(
        [
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.array([22, 36], dtype=np.int64)),
                ak.contents.RecordArray(
                    [
                        ak.contents.NumpyArray(
                            np.array([True, False, True, False, True, False, True, False, True, False, True, False, False, False,
                             False, False, False, False, False, False, False, False, True, False, True, False, True,
                             False, True, False, False, False, False, False, False, False], dtype=np.bool_)
                        ),
                        ak.contents.NumpyArray(
                            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int64)
                        )
                    ],
                    ["part_of_iterative_splitting", "parent_splitting_index"],
                ),
            )
        ],
        ["subjets"],
    )
    # fmt: on

    array = ak.Array(layout)
    result = array.subjets.parent_splitting_index[
        array.subjets.part_of_iterative_splitting
    ]
    assert ak.almost_equal(result, [[0, 1, 2, 3]])
