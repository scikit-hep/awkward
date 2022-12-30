# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak


def test():
    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], [], [7, 11, 13]],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [],
        [7 * 29, 11 * 31, 13 * 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [-1], [], [7, 11, 13]],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [-1],
        [],
        [7 * 29, 11 * 31, 13 * 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], [], [7, 11, 13]],
            [[17, 19], [-1], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [-1],
        [],
        [7 * 29, 11 * 31, 13 * 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], [-1], [7, 11, 13]],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [-1],
        [7 * 29, 11 * 31, 13 * 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], [], [7, 11, 13]],
            [[17, 19], [], [-1], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [-1],
        [7 * 29, 11 * 31, 13 * 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [-1], [], [7, 11, 13]],
            [[17, 19], [39], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [-39],
        [],
        [7 * 29, 11 * 31, 13 * 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], [-1], [7, 11, 13]],
            [[17, 19], [], [39], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [-39],
        [7 * 29, 11 * 31, 13 * 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], []],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], []],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3]],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [-1], []],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [-1],
        [],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], []],
            [[17, 19], [-1], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [-1],
        [],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], [-1]],
            [[17, 19], [], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [-1],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], []],
            [[17, 19], [], [-1], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [-1],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [-1], []],
            [[17, 19], [39], [], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [-39],
        [],
        [29, 31, 37],
    ]

    ak_array = ak.highlevel.Array(
        [
            [[2, 3], [], [-1]],
            [[17, 19], [], [39], [29, 31, 37]],
        ]
    )
    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19],
        [],
        [-39],
        [29, 31, 37],
    ]


def test_other_axis_values():
    ak_array = ak.highlevel.Array(
        [
            [[2, 3, 5], [], [], [7, 11, 13]],
            [[17, 19, 23], [], [], [29, 31, 37]],
        ]
    )

    assert ak.operations.prod(ak_array, axis=-1).to_list() == [
        [2 * 3 * 5, 1, 1, 7 * 11 * 13],
        [17 * 19 * 23, 1, 1, 29 * 31 * 37],
    ]

    assert ak.operations.prod(ak_array, axis=-2).to_list() == [
        [2 * 7, 3 * 11, 5 * 13],
        [17 * 29, 19 * 31, 23 * 37],
    ]

    assert ak.operations.prod(ak_array, axis=-3).to_list() == [
        [2 * 17, 3 * 19, 5 * 23],
        [],
        [],
        [7 * 29, 11 * 31, 13 * 37],
    ]


def test_actual_issue():
    ak_array = ak.highlevel.Array(
        [[[1, 2, 3], [], [4, 3, 2]], [[4, 5, 6], [], [2, 3, 4]]]
    )
    assert ak.operations.min(ak_array, axis=0).to_list() == [
        [1, 2, 3],
        [],
        [2, 3, 2],
    ]
