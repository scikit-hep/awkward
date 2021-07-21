# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_varnewaxis_1():
    array = ak.Array(
        [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        ]
    )
    slicer = ak.Array(
        [[[3, 4], [3, 4], [3, 4]], [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]
    )
    assert array[slicer].tolist() == [
        [[3, 4], [8, 9], [13, 14]],
        [[15, 16, 17, 18], [20, 21, 22, 23], [25, 26, 27, 28]],
    ]


def test_varnewaxis_2():
    array = ak.Array(
        [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        ]
    )
    slicer = ak.Array(
        [[[3, 4], [3, 4], [3, 4]], [[0, 1, None, 3], [0, 1, None, 3], [0, 1, None, 3]]]
    )
    assert array[slicer].tolist() == [
        [[3, 4], [8, 9], [13, 14]],
        [[15, 16, None, 18], [20, 21, None, 23], [25, 26, None, 28]],
    ]


def test_varnewaxis_3():
    array = ak.Array(
        [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        ]
    )
    slicer = ak.Array(
        [
            [
                [False, False, False, True, True],
                [False, False, False, True, True],
                [False, False, False, True, True],
            ],
            [
                [True, True, True, True, False],
                [True, True, True, True, False],
                [True, True, True, True, False],
            ],
        ]
    )
    assert array[slicer].tolist() == [
        [[3, 4], [8, 9], [13, 14]],
        [[15, 16, 17, 18], [20, 21, 22, 23], [25, 26, 27, 28]],
    ]


def test_varnewaxis_4():
    array = ak.Array(
        [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        ]
    )
    slicer = ak.Array(
        [
            [
                [False, False, False, True, True],
                [False, False, False, True, True],
                [False, False, False, True, True],
            ],
            [
                [True, True, None, True, False],
                [True, True, None, True, False],
                [True, True, None, True, False],
            ],
        ]
    )
    assert array[slicer].tolist() == [
        [[3, 4], [8, 9], [13, 14]],
        [[15, 16, None, 18], [20, 21, None, 23], [25, 26, None, 28]],
    ]


def test_varnewaxis_5():
    array = ak.Array(
        [
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        ]
    )
    array = array[[1, 0]]
    slicer = ak.Array(
        [[[3, 4], [3, 4], [3, 4]], [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]
    )
    assert array[slicer].tolist() == [
        [[18, 19], [23, 24], [28, 29]],
        [[0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13]],
    ]


def test_varnewaxis_6():
    array = ak.Array(
        np.array(
            [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
            ]
        )
    )
    slicer = ak.Array(
        [[[3, 4], [3, 4], [3, 4]], [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]
    )
    assert array[slicer].tolist() == [
        [[3, 4], [8, 9], [13, 14]],
        [[15, 16, 17, 18], [20, 21, 22, 23], [25, 26, 27, 28]],
    ]
