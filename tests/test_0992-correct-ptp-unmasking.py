# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    array = ak.highlevel.Array(
        [
            [0, 1, None],
            [None, 3],
            [],
            None,
            [4, 5, None, 6],
        ]
    )

    assert ak.operations.sum(array, axis=-1).to_list() == [
        1,  # 0 + 1
        3,  # 3
        0,  # list is empty
        None,  # list is missing
        15,  # 4 + 5 + 6
    ]

    assert ak.operations.sum(array, axis=-2).to_list() == [
        4,
        9,
        0,
        6,
        # 0+4
        #     1+3+5
        #         no data
        #                     6
    ]

    assert ak.operations.min(array, axis=-1).to_list() == [
        0,  # min([0, 1])
        3,  # min([3])
        None,  # list is empty
        None,  # list is missing
        4,  # min([4, 5, 6])
    ]

    assert ak.operations.min(array, axis=-2).to_list() == [
        0,
        1,
        None,
        6,
        # 0,4
        #     1,3,5
        #          no data
        #                     6
    ]

    # first bug-fix: single '?'
    assert str(ak.operations.min(array, axis=-1).type) == "5 * ?int64"

    # second bug-fix: correct mask_identity=False behavior
    assert ak.operations.ptp(array, axis=-1).to_list() == [
        1,
        0,
        None,
        None,
        2,
    ]
    assert ak.operations.ptp(array, axis=-1, mask_identity=False).to_list() == [
        1,
        0,
        0,
        None,
        2,
    ]

    assert ak.operations.ptp(array, axis=-2).to_list() == [4, 4, None, 0]
    assert ak.operations.ptp(array, axis=-2, mask_identity=False).to_list() == [
        4,
        4,
        0,
        0,
    ]
