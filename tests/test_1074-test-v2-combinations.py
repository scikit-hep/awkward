# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401
from awkward._v2.tmp_for_testing import v1_to_v2, v2_to_v1


def test_ListOffsetArray():
    array = ak.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )
    v2_array = v1_to_v2(array.layout)
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, replacement=False))) == [
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
    assert ak.to_list(
        v2_to_v1(v2_array.combinations(2, replacement=False, keys=["x", "y"]))
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
    tmp = v2_to_v1(
        v2_array.combinations(2, replacement=False, parameters={"some": "param"})
    )
    if isinstance(tmp, ak.partition.PartitionedArray):
        assert ak.partition.first(tmp).content.parameters["some"] == "param"
    else:
        assert tmp.content.parameters["some"] == "param"

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, replacement=True))) == [
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

    assert ak.to_list(v2_to_v1(v2_array.combinations(3, replacement=False))) == [
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

    assert ak.to_list(v2_to_v1(v2_array.combinations(3, replacement=True))) == [
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
    array = ak.Array(np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]]))
    v2_array = v1_to_v2(array.layout)

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, replacement=False))) == [
        [(0.0, 1.1), (0.0, 2.2), (0.0, 3.3), (1.1, 2.2), (1.1, 3.3), (2.2, 3.3)],
        [(4.4, 5.5), (4.4, 6.6), (4.4, 7.7), (5.5, 6.6), (5.5, 7.7), (6.6, 7.7)],
    ]
    assert ak.to_list(
        v2_to_v1(v2_array.combinations(2, replacement=False, keys=["x", "y"]))
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

    tmp = v2_to_v1(
        v2_array.combinations(2, replacement=False, parameters={"some": "param"})
    )
    if isinstance(tmp, ak.partition.PartitionedArray):
        assert ak.partition.first(tmp).content.parameters["some"] == "param"
    else:
        assert tmp.content.parameters["some"] == "param"

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, replacement=True))) == [
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

    assert ak.to_list(v2_to_v1(v2_array.combinations(3, replacement=False))) == [
        [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
        [(4.4, 5.5, 6.6), (4.4, 5.5, 7.7), (4.4, 6.6, 7.7), (5.5, 6.6, 7.7)],
    ]

    assert ak.to_list(v2_to_v1(v2_array.combinations(3, replacement=True))) == [
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
    array = ak.Array([0.0, 1.1, 2.2, 3.3])
    v2_array = v1_to_v2(array.layout)

    assert ak.to_list(
        v2_to_v1(v2_array.combinations(2, replacement=False, axis=0))
    ) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]

    assert ak.to_list(
        v2_to_v1(v2_array.combinations(2, replacement=False, axis=0, keys=["x", "y"]))
    ) == [
        {"x": 0.0, "y": 1.1},
        {"x": 0.0, "y": 2.2},
        {"x": 0.0, "y": 3.3},
        {"x": 1.1, "y": 2.2},
        {"x": 1.1, "y": 3.3},
        {"x": 2.2, "y": 3.3},
    ]

    assert (
        v2_to_v1(
            v2_array.combinations(
                2, replacement=False, axis=0, parameters={"some": "param"}
            )
        ).parameters["some"]
        == "param"
    )

    assert ak.to_list(
        v2_to_v1(v2_array.combinations(3, replacement=False, axis=0))
    ) == [
        (0.0, 1.1, 2.2),
        (0.0, 1.1, 3.3),
        (0.0, 2.2, 3.3),
        (1.1, 2.2, 3.3),
    ]


def test_IndexedArray():
    array = ak.Array(
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
    v2_array = v1_to_v2(array.layout)

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, replacement=False))) == [
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
    array = ak.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    )
    v2_array = v1_to_v2(array.layout)

    assert ak.to_list(
        v2_to_v1(v2_array.combinations(2, axis=1, replacement=False))
    ) == [
        [
            ([0.0, 1.1, 2.2, 3.3], []),
            ([0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6]),
            ([], [4.4, 5.5, 6.6]),
        ],
        [],
        [([7.7], [8.8, 9.9, 10.0, 11.1, 12.2])],
    ]
    assert ak.to_list(
        v2_to_v1(v2_array.combinations(2, axis=2, replacement=False))
    ) == [
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


def test_ByteMaskedArray():
    content = ak.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    v2_array = v1_to_v2(array.layout)

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=0))) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=-3))) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=1))) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=-2))) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=2))) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=-1))) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]


def test_IndexedOptionArray():
    content = ak.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    index = ak.layout.Index64(np.array([0, 1, -1, -1, 4], dtype=np.int64))
    array = ak.Array(ak.layout.IndexedOptionArray64(index, content))
    v2_array = v1_to_v2(array.layout)

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=0))) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=-3))) == [
        ([[0, 1, 2], [], [3, 4]], []),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], None),
        ([[0, 1, 2], [], [3, 4]], [[], [10, 11, 12]]),
        ([], None),
        ([], None),
        ([], [[], [10, 11, 12]]),
        (None, None),
        (None, [[], [10, 11, 12]]),
        (None, [[], [10, 11, 12]]),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=1))) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=-2))) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=2))) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=-1))) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]


def test_NumpyArray():
    v2_array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
    )

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=0))) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=-1))) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(3, axis=-1))) == [
        (0.0, 1.1, 2.2),
        (0.0, 1.1, 3.3),
        (0.0, 2.2, 3.3),
        (1.1, 2.2, 3.3),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(4, axis=-1))) == [
        (0.0, 1.1, 2.2, 3.3)
    ]


def test_BitMaskedArray():
    v2_array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    dtype=np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=0))) == [
        (0.0, 1.0),
        (0.0, 2.0),
        (0.0, 3.0),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, 1.1),
        (0.0, None),
        (0.0, 3.3),
        (0.0, None),
        (0.0, 5.5),
        (1.0, 2.0),
        (1.0, 3.0),
        (1.0, None),
        (1.0, None),
        (1.0, None),
        (1.0, None),
        (1.0, 1.1),
        (1.0, None),
        (1.0, 3.3),
        (1.0, None),
        (1.0, 5.5),
        (2.0, 3.0),
        (2.0, None),
        (2.0, None),
        (2.0, None),
        (2.0, None),
        (2.0, 1.1),
        (2.0, None),
        (2.0, 3.3),
        (2.0, None),
        (2.0, 5.5),
        (3.0, None),
        (3.0, None),
        (3.0, None),
        (3.0, None),
        (3.0, 1.1),
        (3.0, None),
        (3.0, 3.3),
        (3.0, None),
        (3.0, 5.5),
        (None, None),
        (None, None),
        (None, None),
        (None, 1.1),
        (None, None),
        (None, 3.3),
        (None, None),
        (None, 5.5),
        (None, None),
        (None, None),
        (None, 1.1),
        (None, None),
        (None, 3.3),
        (None, None),
        (None, 5.5),
        (None, None),
        (None, 1.1),
        (None, None),
        (None, 3.3),
        (None, None),
        (None, 5.5),
        (None, 1.1),
        (None, None),
        (None, 3.3),
        (None, None),
        (None, 5.5),
        (1.1, None),
        (1.1, 3.3),
        (1.1, None),
        (1.1, 5.5),
        (None, 3.3),
        (None, None),
        (None, 5.5),
        (3.3, None),
        (3.3, 5.5),
        (None, 5.5),
    ]
    assert ak.to_list(v2_to_v1(v2_array.combinations(3, axis=0))) == [
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 3.0),
        (0.0, 1.0, None),
        (0.0, 1.0, None),
        (0.0, 1.0, None),
        (0.0, 1.0, None),
        (0.0, 1.0, 1.1),
        (0.0, 1.0, None),
        (0.0, 1.0, 3.3),
        (0.0, 1.0, None),
        (0.0, 1.0, 5.5),
        (0.0, 2.0, 3.0),
        (0.0, 2.0, None),
        (0.0, 2.0, None),
        (0.0, 2.0, None),
        (0.0, 2.0, None),
        (0.0, 2.0, 1.1),
        (0.0, 2.0, None),
        (0.0, 2.0, 3.3),
        (0.0, 2.0, None),
        (0.0, 2.0, 5.5),
        (0.0, 3.0, None),
        (0.0, 3.0, None),
        (0.0, 3.0, None),
        (0.0, 3.0, None),
        (0.0, 3.0, 1.1),
        (0.0, 3.0, None),
        (0.0, 3.0, 3.3),
        (0.0, 3.0, None),
        (0.0, 3.0, 5.5),
        (0.0, None, None),
        (0.0, None, None),
        (0.0, None, None),
        (0.0, None, 1.1),
        (0.0, None, None),
        (0.0, None, 3.3),
        (0.0, None, None),
        (0.0, None, 5.5),
        (0.0, None, None),
        (0.0, None, None),
        (0.0, None, 1.1),
        (0.0, None, None),
        (0.0, None, 3.3),
        (0.0, None, None),
        (0.0, None, 5.5),
        (0.0, None, None),
        (0.0, None, 1.1),
        (0.0, None, None),
        (0.0, None, 3.3),
        (0.0, None, None),
        (0.0, None, 5.5),
        (0.0, None, 1.1),
        (0.0, None, None),
        (0.0, None, 3.3),
        (0.0, None, None),
        (0.0, None, 5.5),
        (0.0, 1.1, None),
        (0.0, 1.1, 3.3),
        (0.0, 1.1, None),
        (0.0, 1.1, 5.5),
        (0.0, None, 3.3),
        (0.0, None, None),
        (0.0, None, 5.5),
        (0.0, 3.3, None),
        (0.0, 3.3, 5.5),
        (0.0, None, 5.5),
        (1.0, 2.0, 3.0),
        (1.0, 2.0, None),
        (1.0, 2.0, None),
        (1.0, 2.0, None),
        (1.0, 2.0, None),
        (1.0, 2.0, 1.1),
        (1.0, 2.0, None),
        (1.0, 2.0, 3.3),
        (1.0, 2.0, None),
        (1.0, 2.0, 5.5),
        (1.0, 3.0, None),
        (1.0, 3.0, None),
        (1.0, 3.0, None),
        (1.0, 3.0, None),
        (1.0, 3.0, 1.1),
        (1.0, 3.0, None),
        (1.0, 3.0, 3.3),
        (1.0, 3.0, None),
        (1.0, 3.0, 5.5),
        (1.0, None, None),
        (1.0, None, None),
        (1.0, None, None),
        (1.0, None, 1.1),
        (1.0, None, None),
        (1.0, None, 3.3),
        (1.0, None, None),
        (1.0, None, 5.5),
        (1.0, None, None),
        (1.0, None, None),
        (1.0, None, 1.1),
        (1.0, None, None),
        (1.0, None, 3.3),
        (1.0, None, None),
        (1.0, None, 5.5),
        (1.0, None, None),
        (1.0, None, 1.1),
        (1.0, None, None),
        (1.0, None, 3.3),
        (1.0, None, None),
        (1.0, None, 5.5),
        (1.0, None, 1.1),
        (1.0, None, None),
        (1.0, None, 3.3),
        (1.0, None, None),
        (1.0, None, 5.5),
        (1.0, 1.1, None),
        (1.0, 1.1, 3.3),
        (1.0, 1.1, None),
        (1.0, 1.1, 5.5),
        (1.0, None, 3.3),
        (1.0, None, None),
        (1.0, None, 5.5),
        (1.0, 3.3, None),
        (1.0, 3.3, 5.5),
        (1.0, None, 5.5),
        (2.0, 3.0, None),
        (2.0, 3.0, None),
        (2.0, 3.0, None),
        (2.0, 3.0, None),
        (2.0, 3.0, 1.1),
        (2.0, 3.0, None),
        (2.0, 3.0, 3.3),
        (2.0, 3.0, None),
        (2.0, 3.0, 5.5),
        (2.0, None, None),
        (2.0, None, None),
        (2.0, None, None),
        (2.0, None, 1.1),
        (2.0, None, None),
        (2.0, None, 3.3),
        (2.0, None, None),
        (2.0, None, 5.5),
        (2.0, None, None),
        (2.0, None, None),
        (2.0, None, 1.1),
        (2.0, None, None),
        (2.0, None, 3.3),
        (2.0, None, None),
        (2.0, None, 5.5),
        (2.0, None, None),
        (2.0, None, 1.1),
        (2.0, None, None),
        (2.0, None, 3.3),
        (2.0, None, None),
        (2.0, None, 5.5),
        (2.0, None, 1.1),
        (2.0, None, None),
        (2.0, None, 3.3),
        (2.0, None, None),
        (2.0, None, 5.5),
        (2.0, 1.1, None),
        (2.0, 1.1, 3.3),
        (2.0, 1.1, None),
        (2.0, 1.1, 5.5),
        (2.0, None, 3.3),
        (2.0, None, None),
        (2.0, None, 5.5),
        (2.0, 3.3, None),
        (2.0, 3.3, 5.5),
        (2.0, None, 5.5),
        (3.0, None, None),
        (3.0, None, None),
        (3.0, None, None),
        (3.0, None, 1.1),
        (3.0, None, None),
        (3.0, None, 3.3),
        (3.0, None, None),
        (3.0, None, 5.5),
        (3.0, None, None),
        (3.0, None, None),
        (3.0, None, 1.1),
        (3.0, None, None),
        (3.0, None, 3.3),
        (3.0, None, None),
        (3.0, None, 5.5),
        (3.0, None, None),
        (3.0, None, 1.1),
        (3.0, None, None),
        (3.0, None, 3.3),
        (3.0, None, None),
        (3.0, None, 5.5),
        (3.0, None, 1.1),
        (3.0, None, None),
        (3.0, None, 3.3),
        (3.0, None, None),
        (3.0, None, 5.5),
        (3.0, 1.1, None),
        (3.0, 1.1, 3.3),
        (3.0, 1.1, None),
        (3.0, 1.1, 5.5),
        (3.0, None, 3.3),
        (3.0, None, None),
        (3.0, None, 5.5),
        (3.0, 3.3, None),
        (3.0, 3.3, 5.5),
        (3.0, None, 5.5),
        (None, None, None),
        (None, None, None),
        (None, None, 1.1),
        (None, None, None),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, None, None),
        (None, None, 1.1),
        (None, None, None),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, None, 1.1),
        (None, None, None),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, 1.1, None),
        (None, 1.1, 3.3),
        (None, 1.1, None),
        (None, 1.1, 5.5),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, 3.3, None),
        (None, 3.3, 5.5),
        (None, None, 5.5),
        (None, None, None),
        (None, None, 1.1),
        (None, None, None),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, None, 1.1),
        (None, None, None),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, 1.1, None),
        (None, 1.1, 3.3),
        (None, 1.1, None),
        (None, 1.1, 5.5),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, 3.3, None),
        (None, 3.3, 5.5),
        (None, None, 5.5),
        (None, None, 1.1),
        (None, None, None),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, 1.1, None),
        (None, 1.1, 3.3),
        (None, 1.1, None),
        (None, 1.1, 5.5),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, 3.3, None),
        (None, 3.3, 5.5),
        (None, None, 5.5),
        (None, 1.1, None),
        (None, 1.1, 3.3),
        (None, 1.1, None),
        (None, 1.1, 5.5),
        (None, None, 3.3),
        (None, None, None),
        (None, None, 5.5),
        (None, 3.3, None),
        (None, 3.3, 5.5),
        (None, None, 5.5),
        (1.1, None, 3.3),
        (1.1, None, None),
        (1.1, None, 5.5),
        (1.1, 3.3, None),
        (1.1, 3.3, 5.5),
        (1.1, None, 5.5),
        (None, 3.3, None),
        (None, 3.3, 5.5),
        (None, None, 5.5),
        (3.3, None, 5.5),
    ]


def test_EmptyArray():
    v2_array = ak._v2.contents.emptyarray.EmptyArray()

    assert ak.to_list(v2_to_v1(v2_array.combinations(2, axis=0))) == []
