# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.convert.to_list


def test_ListOffsetArray():
    array = ak._v2.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )

    assert to_list(ak._v2.operations.structure.combinations(array, 2, replacement=False)) == [
        [(0.0, 1.1), (0.0, 2.2), (0.0, 3.3), (1.1, 2.2), (1.1, 3.3), (2.2, 3.3)],
        [],
        [(4.4, 5.5), (4.4, 6.6), (5.5, 6.6)],
        [],
        [
            (8.8, 9.9),
            (8.8, 10.0),
            (8.8, 11.1),
            (8.8, 12.2),
            (9.9, 10.0),
            (9.9, 11.1),
            (9.9, 12.2),
            (10.0, 11.1),
            (10.0, 12.2),
            (11.1, 12.2),
        ],
    ]
    assert to_list(
        ak._v2.operations.structure.combinations(array, 2, replacement=False, fields=["x", "y"])
    ) == [
        [
            {"x": 0.0, "y": 1.1},
            {"x": 0.0, "y": 2.2},
            {"x": 0.0, "y": 3.3},
            {"x": 1.1, "y": 2.2},
            {"x": 1.1, "y": 3.3},
            {"x": 2.2, "y": 3.3},
        ],
        [],
        [{"x": 4.4, "y": 5.5}, {"x": 4.4, "y": 6.6}, {"x": 5.5, "y": 6.6}],
        [],
        [
            {"x": 8.8, "y": 9.9},
            {"x": 8.8, "y": 10.0},
            {"x": 8.8, "y": 11.1},
            {"x": 8.8, "y": 12.2},
            {"x": 9.9, "y": 10.0},
            {"x": 9.9, "y": 11.1},
            {"x": 9.9, "y": 12.2},
            {"x": 10.0, "y": 11.1},
            {"x": 10.0, "y": 12.2},
            {"x": 11.1, "y": 12.2},
        ],
    ]
    tmp = ak._v2.operations.structure.combinations(
        array, 2, replacement=False, parameters={"some": "param"}
    ).layout

    assert tmp.content.parameters["some"] == "param"

    assert to_list(ak._v2.operations.structure.combinations(array, 2, replacement=True)) == [
        [
            (0.0, 0.0),
            (0.0, 1.1),
            (0.0, 2.2),
            (0.0, 3.3),
            (1.1, 1.1),
            (1.1, 2.2),
            (1.1, 3.3),
            (2.2, 2.2),
            (2.2, 3.3),
            (3.3, 3.3),
        ],
        [],
        [(4.4, 4.4), (4.4, 5.5), (4.4, 6.6), (5.5, 5.5), (5.5, 6.6), (6.6, 6.6)],
        [(7.7, 7.7)],
        [
            (8.8, 8.8),
            (8.8, 9.9),
            (8.8, 10.0),
            (8.8, 11.1),
            (8.8, 12.2),
            (9.9, 9.9),
            (9.9, 10.0),
            (9.9, 11.1),
            (9.9, 12.2),
            (10.0, 10.0),
            (10.0, 11.1),
            (10.0, 12.2),
            (11.1, 11.1),
            (11.1, 12.2),
            (12.2, 12.2),
        ],
    ]

    assert to_list(ak._v2.operations.structure.combinations(array, 3, replacement=False)) == [
        [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
        [],
        [(4.4, 5.5, 6.6)],
        [],
        [
            (8.8, 9.9, 10.0),
            (8.8, 9.9, 11.1),
            (8.8, 9.9, 12.2),
            (8.8, 10.0, 11.1),
            (8.8, 10.0, 12.2),
            (8.8, 11.1, 12.2),
            (9.9, 10.0, 11.1),
            (9.9, 10.0, 12.2),
            (9.9, 11.1, 12.2),
            (10.0, 11.1, 12.2),
        ],
    ]

    assert to_list(ak._v2.operations.structure.combinations(array, 3, replacement=True)) == [
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.1),
            (0.0, 0.0, 2.2),
            (0.0, 0.0, 3.3),
            (0.0, 1.1, 1.1),
            (0.0, 1.1, 2.2),
            (0.0, 1.1, 3.3),
            (0.0, 2.2, 2.2),
            (0.0, 2.2, 3.3),
            (0.0, 3.3, 3.3),
            (1.1, 1.1, 1.1),
            (1.1, 1.1, 2.2),
            (1.1, 1.1, 3.3),
            (1.1, 2.2, 2.2),
            (1.1, 2.2, 3.3),
            (1.1, 3.3, 3.3),
            (2.2, 2.2, 2.2),
            (2.2, 2.2, 3.3),
            (2.2, 3.3, 3.3),
            (3.3, 3.3, 3.3),
        ],
        [],
        [
            (4.4, 4.4, 4.4),
            (4.4, 4.4, 5.5),
            (4.4, 4.4, 6.6),
            (4.4, 5.5, 5.5),
            (4.4, 5.5, 6.6),
            (4.4, 6.6, 6.6),
            (5.5, 5.5, 5.5),
            (5.5, 5.5, 6.6),
            (5.5, 6.6, 6.6),
            (6.6, 6.6, 6.6),
        ],
        [(7.7, 7.7, 7.7)],
        [
            (8.8, 8.8, 8.8),
            (8.8, 8.8, 9.9),
            (8.8, 8.8, 10.0),
            (8.8, 8.8, 11.1),
            (8.8, 8.8, 12.2),
            (8.8, 9.9, 9.9),
            (8.8, 9.9, 10.0),
            (8.8, 9.9, 11.1),
            (8.8, 9.9, 12.2),
            (8.8, 10.0, 10.0),
            (8.8, 10.0, 11.1),
            (8.8, 10.0, 12.2),
            (8.8, 11.1, 11.1),
            (8.8, 11.1, 12.2),
            (8.8, 12.2, 12.2),
            (9.9, 9.9, 9.9),
            (9.9, 9.9, 10.0),
            (9.9, 9.9, 11.1),
            (9.9, 9.9, 12.2),
            (9.9, 10.0, 10.0),
            (9.9, 10.0, 11.1),
            (9.9, 10.0, 12.2),
            (9.9, 11.1, 11.1),
            (9.9, 11.1, 12.2),
            (9.9, 12.2, 12.2),
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 11.1),
            (10.0, 10.0, 12.2),
            (10.0, 11.1, 11.1),
            (10.0, 11.1, 12.2),
            (10.0, 12.2, 12.2),
            (11.1, 11.1, 11.1),
            (11.1, 11.1, 12.2),
            (11.1, 12.2, 12.2),
            (12.2, 12.2, 12.2),
        ],
    ]


def test_RegularArray():
    array = ak._v2.Array(np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]]))

    assert to_list(ak._v2.operations.structure.combinations(array, 2, replacement=False)) == [
        [(0.0, 1.1), (0.0, 2.2), (0.0, 3.3), (1.1, 2.2), (1.1, 3.3), (2.2, 3.3)],
        [(4.4, 5.5), (4.4, 6.6), (4.4, 7.7), (5.5, 6.6), (5.5, 7.7), (6.6, 7.7)],
    ]
    assert to_list(
        ak._v2.operations.structure.combinations(array, 2, replacement=False, fields=["x", "y"])
    ) == [
        [
            {"x": 0.0, "y": 1.1},
            {"x": 0.0, "y": 2.2},
            {"x": 0.0, "y": 3.3},
            {"x": 1.1, "y": 2.2},
            {"x": 1.1, "y": 3.3},
            {"x": 2.2, "y": 3.3},
        ],
        [
            {"x": 4.4, "y": 5.5},
            {"x": 4.4, "y": 6.6},
            {"x": 4.4, "y": 7.7},
            {"x": 5.5, "y": 6.6},
            {"x": 5.5, "y": 7.7},
            {"x": 6.6, "y": 7.7},
        ],
    ]

    tmp = ak._v2.operations.structure.combinations(
        array, 2, replacement=False, parameters={"some": "param"}
    ).layout
    assert tmp.content.parameters["some"] == "param"

    assert to_list(ak._v2.operations.structure.combinations(array, 2, replacement=True)) == [
        [
            (0.0, 0.0),
            (0.0, 1.1),
            (0.0, 2.2),
            (0.0, 3.3),
            (1.1, 1.1),
            (1.1, 2.2),
            (1.1, 3.3),
            (2.2, 2.2),
            (2.2, 3.3),
            (3.3, 3.3),
        ],
        [
            (4.4, 4.4),
            (4.4, 5.5),
            (4.4, 6.6),
            (4.4, 7.7),
            (5.5, 5.5),
            (5.5, 6.6),
            (5.5, 7.7),
            (6.6, 6.6),
            (6.6, 7.7),
            (7.7, 7.7),
        ],
    ]

    assert to_list(ak._v2.operations.structure.combinations(array, 3, replacement=False)) == [
        [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
        [(4.4, 5.5, 6.6), (4.4, 5.5, 7.7), (4.4, 6.6, 7.7), (5.5, 6.6, 7.7)],
    ]

    assert to_list(ak._v2.operations.structure.combinations(array, 3, replacement=True)) == [
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.1),
            (0.0, 0.0, 2.2),
            (0.0, 0.0, 3.3),
            (0.0, 1.1, 1.1),
            (0.0, 1.1, 2.2),
            (0.0, 1.1, 3.3),
            (0.0, 2.2, 2.2),
            (0.0, 2.2, 3.3),
            (0.0, 3.3, 3.3),
            (1.1, 1.1, 1.1),
            (1.1, 1.1, 2.2),
            (1.1, 1.1, 3.3),
            (1.1, 2.2, 2.2),
            (1.1, 2.2, 3.3),
            (1.1, 3.3, 3.3),
            (2.2, 2.2, 2.2),
            (2.2, 2.2, 3.3),
            (2.2, 3.3, 3.3),
            (3.3, 3.3, 3.3),
        ],
        [
            (4.4, 4.4, 4.4),
            (4.4, 4.4, 5.5),
            (4.4, 4.4, 6.6),
            (4.4, 4.4, 7.7),
            (4.4, 5.5, 5.5),
            (4.4, 5.5, 6.6),
            (4.4, 5.5, 7.7),
            (4.4, 6.6, 6.6),
            (4.4, 6.6, 7.7),
            (4.4, 7.7, 7.7),
            (5.5, 5.5, 5.5),
            (5.5, 5.5, 6.6),
            (5.5, 5.5, 7.7),
            (5.5, 6.6, 6.6),
            (5.5, 6.6, 7.7),
            (5.5, 7.7, 7.7),
            (6.6, 6.6, 6.6),
            (6.6, 6.6, 7.7),
            (6.6, 7.7, 7.7),
            (7.7, 7.7, 7.7),
        ],
    ]


def test_axis0():
    array = ak._v2.Array([0.0, 1.1, 2.2, 3.3])

    assert to_list(ak._v2.operations.structure.combinations(array, 2, replacement=False, axis=0)) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]

    assert to_list(
        ak._v2.operations.structure.combinations(array, 2, replacement=False, axis=0, fields=["x", "y"])
    ) == [
        {"x": 0.0, "y": 1.1},
        {"x": 0.0, "y": 2.2},
        {"x": 0.0, "y": 3.3},
        {"x": 1.1, "y": 2.2},
        {"x": 1.1, "y": 3.3},
        {"x": 2.2, "y": 3.3},
    ]

    assert (
        ak._v2.operations.structure.combinations(
            array, 2, replacement=False, axis=0, parameters={"some": "param"}
        ).layout.parameters["some"]
        == "param"
    )

    assert to_list(ak._v2.operations.structure.combinations(array, 3, replacement=False, axis=0)) == [
        (0.0, 1.1, 2.2),
        (0.0, 1.1, 3.3),
        (0.0, 2.2, 3.3),
        (1.1, 2.2, 3.3),
    ]


def test_IndexedArray():
    array = ak._v2.Array(
        [
            [0.0, 1.1, 2.2, 3.3],
            [],
            [4.4, 5.5, 6.6],
            None,
            [7.7],
            None,
            [8.8, 9.9, 10.0, 11.1, 12.2],
        ]
    )

    assert to_list(ak._v2.operations.structure.combinations(array, 2, replacement=False)) == [
        [(0.0, 1.1), (0.0, 2.2), (0.0, 3.3), (1.1, 2.2), (1.1, 3.3), (2.2, 3.3)],
        [],
        [(4.4, 5.5), (4.4, 6.6), (5.5, 6.6)],
        None,
        [],
        None,
        [
            (8.8, 9.9),
            (8.8, 10.0),
            (8.8, 11.1),
            (8.8, 12.2),
            (9.9, 10.0),
            (9.9, 11.1),
            (9.9, 12.2),
            (10.0, 11.1),
            (10.0, 12.2),
            (11.1, 12.2),
        ],
    ]


def test_axis2():
    array = ak._v2.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    )

    assert to_list(ak._v2.operations.structure.combinations(array, 2, axis=1, replacement=False)) == [
        [
            ([0.0, 1.1, 2.2, 3.3], []),
            ([0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6]),
            ([], [4.4, 5.5, 6.6]),
        ],
        [],
        [([7.7], [8.8, 9.9, 10.0, 11.1, 12.2])],
    ]

    assert to_list(ak._v2.operations.structure.combinations(array, 2, axis=2, replacement=False)) == [
        [
            [(0.0, 1.1), (0.0, 2.2), (0.0, 3.3), (1.1, 2.2), (1.1, 3.3), (2.2, 3.3)],
            [],
            [(4.4, 5.5), (4.4, 6.6), (5.5, 6.6)],
        ],
        [],
        [
            [],
            [
                (8.8, 9.9),
                (8.8, 10.0),
                (8.8, 11.1),
                (8.8, 12.2),
                (9.9, 10.0),
                (9.9, 11.1),
                (9.9, 12.2),
                (10.0, 11.1),
                (10.0, 12.2),
                (11.1, 12.2),
            ],
        ],
    ]


def test_argcombinations():
    array = ak._v2.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )

    assert to_list(ak._v2.operations.structure.argcombinations(array, 2, replacement=False)) == [
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        [],
        [(0, 1), (0, 2), (1, 2)],
        [],
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ],
    ]
