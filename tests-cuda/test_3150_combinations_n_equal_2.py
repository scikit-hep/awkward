from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0079_argchoose_and_choose_ListOffsetArray():
    array = ak.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )
    array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.combinations(array, 2, replacement=False)) == [
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
        ak.operations.combinations(array, 2, replacement=False, fields=["x", "y"])
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
    tmp = ak.operations.combinations(
        array, 2, replacement=False, parameters={"some": "param"}
    ).layout

    assert tmp.content.parameters["some"] == "param"

    assert to_list(ak.operations.combinations(array, 2, replacement=True)) == [
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
    del array


def test_0079_argchoose_and_choose_RegularArray():
    array = ak.Array(np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]]))
    array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.combinations(array, 2, replacement=False)) == [
        [(0.0, 1.1), (0.0, 2.2), (0.0, 3.3), (1.1, 2.2), (1.1, 3.3), (2.2, 3.3)],
        [(4.4, 5.5), (4.4, 6.6), (4.4, 7.7), (5.5, 6.6), (5.5, 7.7), (6.6, 7.7)],
    ]
    assert to_list(
        ak.operations.combinations(array, 2, replacement=False, fields=["x", "y"])
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

    tmp = ak.operations.combinations(
        array, 2, replacement=False, parameters={"some": "param"}
    ).layout
    assert tmp.content.parameters["some"] == "param"

    assert to_list(ak.operations.combinations(array, 2, replacement=True)) == [
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
    del array


def test_0079_argchoose_and_choose_axis0():
    array = ak.Array([0.0, 1.1, 2.2, 3.3])
    array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.combinations(array, 2, replacement=False, axis=0)) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]

    assert to_list(
        ak.operations.combinations(
            array, 2, replacement=False, axis=0, fields=["x", "y"]
        )
    ) == [
        {"x": 0.0, "y": 1.1},
        {"x": 0.0, "y": 2.2},
        {"x": 0.0, "y": 3.3},
        {"x": 1.1, "y": 2.2},
        {"x": 1.1, "y": 3.3},
        {"x": 2.2, "y": 3.3},
    ]

    assert (
        ak.operations.combinations(
            array, 2, replacement=False, axis=0, parameters={"some": "param"}
        ).layout.parameters["some"]
        == "param"
    )
    del array


def test_0079_argchoose_and_choose_IndexedArray():
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
    array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.combinations(array, 2, replacement=False)) == [
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
    del array


def test_0079_argchoose_and_choose_axis2():
    array = ak.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    )
    array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.combinations(array, 2, axis=1, replacement=False)) == [
        [
            ([0.0, 1.1, 2.2, 3.3], []),
            ([0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6]),
            ([], [4.4, 5.5, 6.6]),
        ],
        [],
        [([7.7], [8.8, 9.9, 10.0, 11.1, 12.2])],
    ]

    assert to_list(ak.operations.combinations(array, 2, axis=2, replacement=False)) == [
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
    del array


def test_0079_argchoose_and_choose_argcombinations():
    array = ak.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )
    array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.argcombinations(array, 2, replacement=False)) == [
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
    del array


def test_1074_combinations_ListOffsetArray():
    v2_array = ak.highlevel.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    ).layout
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, replacement=False)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, replacement=False).form
        == ak._do.combinations(v2_array, 2, replacement=False).form
    )
    assert to_list(
        ak._do.combinations(v2_array, 2, replacement=False, fields=["x", "y"])
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
    assert (
        ak._do.combinations(
            v2_array.to_typetracer(), 2, replacement=False, fields=["x", "y"]
        ).form
        == ak._do.combinations(v2_array, 2, replacement=False, fields=["x", "y"]).form
    )

    assert (
        ak._do.combinations(
            v2_array, 2, replacement=False, parameters={"some": "param"}
        ).content.parameters["some"]
        == "param"
    )
    assert (
        ak._do.combinations(
            v2_array.to_typetracer(), 2, replacement=False, parameters={"some": "param"}
        ).form
        == ak._do.combinations(
            v2_array, 2, replacement=False, parameters={"some": "param"}
        ).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, replacement=True)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, replacement=True).form
        == ak._do.combinations(v2_array, 2, replacement=True).form
    )
    del v2_array


def test_1074_combinations_RegularArray():
    v2_array = ak.highlevel.Array(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]])
    ).layout
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, replacement=False)) == [
        [(0.0, 1.1), (0.0, 2.2), (0.0, 3.3), (1.1, 2.2), (1.1, 3.3), (2.2, 3.3)],
        [(4.4, 5.5), (4.4, 6.6), (4.4, 7.7), (5.5, 6.6), (5.5, 7.7), (6.6, 7.7)],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, replacement=False).form
        == ak._do.combinations(v2_array, 2, replacement=False).form
    )

    assert to_list(
        ak._do.combinations(v2_array, 2, replacement=False, fields=["x", "y"])
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
    assert (
        ak._do.combinations(
            v2_array.to_typetracer(), 2, replacement=False, fields=["x", "y"]
        ).form
        == ak._do.combinations(v2_array, 2, replacement=False, fields=["x", "y"]).form
    )

    assert (
        ak._do.combinations(
            v2_array, 2, replacement=False, parameters={"some": "param"}
        ).content.parameters["some"]
        == "param"
    )
    assert (
        ak._do.combinations(
            v2_array.to_typetracer(), 2, replacement=False, parameters={"some": "param"}
        ).form
        == ak._do.combinations(
            v2_array, 2, replacement=False, parameters={"some": "param"}
        ).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, replacement=True)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, replacement=True).form
        == ak._do.combinations(v2_array, 2, replacement=True).form
    )
    del v2_array


def test_1074_combinations_axis0():
    v2_array = ak.highlevel.Array([0.0, 1.1, 2.2, 3.3]).layout
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, replacement=False, axis=0)) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, replacement=False, axis=0).form
        == ak._do.combinations(v2_array, 2, replacement=False, axis=0).form
    )

    assert to_list(
        ak._do.combinations(v2_array, 2, replacement=False, axis=0, fields=["x", "y"])
    ) == [
        {"x": 0.0, "y": 1.1},
        {"x": 0.0, "y": 2.2},
        {"x": 0.0, "y": 3.3},
        {"x": 1.1, "y": 2.2},
        {"x": 1.1, "y": 3.3},
        {"x": 2.2, "y": 3.3},
    ]
    assert (
        ak._do.combinations(
            v2_array.to_typetracer(), 2, replacement=False, axis=0, fields=["x", "y"]
        ).form
        == ak._do.combinations(
            v2_array, 2, replacement=False, axis=0, fields=["x", "y"]
        ).form
    )

    assert (
        ak._do.combinations(
            v2_array, 2, replacement=False, axis=0, parameters={"some": "param"}
        ).parameters["some"]
        == "param"
    )
    assert (
        ak._do.combinations(
            v2_array.to_typetracer(),
            2,
            replacement=False,
            axis=0,
            parameters={"some": "param"},
        ).form
        == ak._do.combinations(
            v2_array, 2, replacement=False, axis=0, parameters={"some": "param"}
        ).form
    )
    del v2_array


def test_1074_combinations_IndexedArray():
    v2_array = ak.highlevel.Array(
        [
            [0.0, 1.1, 2.2, 3.3],
            [],
            [4.4, 5.5, 6.6],
            None,
            [7.7],
            None,
            [8.8, 9.9, 10.0, 11.1, 12.2],
        ]
    ).layout
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, replacement=False)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, replacement=False).form
        == ak._do.combinations(v2_array, 2, replacement=False).form
    )
    del v2_array


def test_1074_combinations_axis2():
    v2_array = ak.highlevel.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    ).layout
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=1, replacement=False)) == [
        [
            ([0.0, 1.1, 2.2, 3.3], []),
            ([0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6]),
            ([], [4.4, 5.5, 6.6]),
        ],
        [],
        [([7.7], [8.8, 9.9, 10.0, 11.1, 12.2])],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=1, replacement=False).form
        == ak._do.combinations(v2_array, 2, axis=1, replacement=False).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=2, replacement=False)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=2, replacement=False).form
        == ak._do.combinations(v2_array, 2, axis=2, replacement=False).form
    )
    del v2_array


def test_1074_combinations_ByteMaskedArray():
    content = ak.operations.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v2_array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-3)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-3).form
        == ak._do.combinations(v2_array, 2, axis=-3).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=1)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=1).form
        == ak._do.combinations(v2_array, 2, axis=1).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-2)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-2).form
        == ak._do.combinations(v2_array, 2, axis=-2).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=2)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=2).form
        == ak._do.combinations(v2_array, 2, axis=2).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-1)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-1).form
        == ak._do.combinations(v2_array, 2, axis=-1).form
    )
    del v2_array


def test_1074_combinations_IndexedOptionArray():
    content = ak.operations.from_iter(
        [[[0, 1, 2], [], [3, 4]], [], [[5]], [[6, 7, 8, 9]], [[], [10, 11, 12]]],
        highlevel=False,
    )
    index = ak.index.Index64(np.array([0, 1, -1, -1, 4], dtype=np.int64))
    v2_array = ak.contents.IndexedOptionArray(index, content)
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-3)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-3).form
        == ak._do.combinations(v2_array, 2, axis=-3).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=1)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=1).form
        == ak._do.combinations(v2_array, 2, axis=1).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-2)) == [
        [([0, 1, 2], []), ([0, 1, 2], [3, 4]), ([], [3, 4])],
        [],
        None,
        None,
        [([], [10, 11, 12])],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-2).form
        == ak._do.combinations(v2_array, 2, axis=-2).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=2)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=2).form
        == ak._do.combinations(v2_array, 2, axis=2).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-1)) == [
        [[(0, 1), (0, 2), (1, 2)], [], [(3, 4)]],
        [],
        None,
        None,
        [[], [(10, 11), (10, 12), (11, 12)]],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-1).form
        == ak._do.combinations(v2_array, 2, axis=-1).form
    )
    del v2_array


def test_1074_combinations_NumpyArray():
    v2_array = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
    )
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-1)) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-1).form
        == ak._do.combinations(v2_array, 2, axis=-1).form
    )
    del v2_array


def test_1074_combinations_BitMaskedArray():
    v2_array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == [
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
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )
    del v2_array


def test_1074_combinations_EmptyArray():
    v2_array = ak.contents.emptyarray.EmptyArray()
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == []
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )
    del v2_array


def test_1074_combinations_RecordArray():
    v2_array = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1])),
        ak.index.Index(np.array([7, 100, 3, 200])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == [
        ([{"nest": 1.1}, {"nest": 2.2}, {"nest": 3.3}], []),
        ([{"nest": 1.1}, {"nest": 2.2}, {"nest": 3.3}], [{"nest": 4.4}, {"nest": 5.5}]),
        ([], [{"nest": 4.4}, {"nest": 5.5}]),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=1)) == [
        [
            ({"nest": 1.1}, {"nest": 2.2}),
            ({"nest": 1.1}, {"nest": 3.3}),
            ({"nest": 2.2}, {"nest": 3.3}),
        ],
        [],
        [({"nest": 4.4}, {"nest": 5.5})],
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=1).form
        == ak._do.combinations(v2_array, 2, axis=1).form
    )
    del v2_array


def test_1074_combinations_UnionArray():
    v2_array = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.recordarray.RecordArray(
                [ak.from_iter(["1", "2", "3"], highlevel=False)], ["nest"]
            ),
            ak.contents.recordarray.RecordArray(
                [
                    ak.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == [
        ({"nest": 5.5}, {"nest": 4.4}),
        ({"nest": 5.5}, {"nest": "1"}),
        ({"nest": 5.5}, {"nest": "2"}),
        ({"nest": 5.5}, {"nest": 3.3}),
        ({"nest": 5.5}, {"nest": "3"}),
        ({"nest": 5.5}, {"nest": 5.5}),
        ({"nest": 4.4}, {"nest": "1"}),
        ({"nest": 4.4}, {"nest": "2"}),
        ({"nest": 4.4}, {"nest": 3.3}),
        ({"nest": 4.4}, {"nest": "3"}),
        ({"nest": 4.4}, {"nest": 5.5}),
        ({"nest": "1"}, {"nest": "2"}),
        ({"nest": "1"}, {"nest": 3.3}),
        ({"nest": "1"}, {"nest": "3"}),
        ({"nest": "1"}, {"nest": 5.5}),
        ({"nest": "2"}, {"nest": 3.3}),
        ({"nest": "2"}, {"nest": "3"}),
        ({"nest": "2"}, {"nest": 5.5}),
        ({"nest": 3.3}, {"nest": "3"}),
        ({"nest": 3.3}, {"nest": 5.5}),
        ({"nest": "3"}, {"nest": 5.5}),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-1)) == [
        ({"nest": 5.5}, {"nest": 4.4}),
        ({"nest": 5.5}, {"nest": "1"}),
        ({"nest": 5.5}, {"nest": "2"}),
        ({"nest": 5.5}, {"nest": 3.3}),
        ({"nest": 5.5}, {"nest": "3"}),
        ({"nest": 5.5}, {"nest": 5.5}),
        ({"nest": 4.4}, {"nest": "1"}),
        ({"nest": 4.4}, {"nest": "2"}),
        ({"nest": 4.4}, {"nest": 3.3}),
        ({"nest": 4.4}, {"nest": "3"}),
        ({"nest": 4.4}, {"nest": 5.5}),
        ({"nest": "1"}, {"nest": "2"}),
        ({"nest": "1"}, {"nest": 3.3}),
        ({"nest": "1"}, {"nest": "3"}),
        ({"nest": "1"}, {"nest": 5.5}),
        ({"nest": "2"}, {"nest": 3.3}),
        ({"nest": "2"}, {"nest": "3"}),
        ({"nest": "2"}, {"nest": 5.5}),
        ({"nest": 3.3}, {"nest": "3"}),
        ({"nest": 3.3}, {"nest": 5.5}),
        ({"nest": "3"}, {"nest": 5.5}),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-1).form
        == ak._do.combinations(v2_array, 2, axis=-1).form
    )
    del v2_array


def test_1074_combinations_UnmaskedArray():
    v2_array = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )
    v2_array = ak.to_backend(v2_array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(v2_array, 2, axis=0)) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=0).form
        == ak._do.combinations(v2_array, 2, axis=0).form
    )

    assert to_list(ak._do.combinations(v2_array, 2, axis=-1)) == [
        (0.0, 1.1),
        (0.0, 2.2),
        (0.0, 3.3),
        (1.1, 2.2),
        (1.1, 3.3),
        (2.2, 3.3),
    ]
    assert (
        ak._do.combinations(v2_array.to_typetracer(), 2, axis=-1).form
        == ak._do.combinations(v2_array, 2, axis=-1).form
    )
    del v2_array


def test_block_boundary_combinations():
    content = ak.contents.NumpyArray(np.arange(300))
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)

    assert to_list(
        ak.combinations(cuda_content, 2, axis=0, replacement=False)
    ) == to_list(ak.combinations(content, 2, axis=0, replacement=False))
    assert to_list(
        ak.combinations(cuda_content, 2, axis=0, replacement=True)
    ) == to_list(ak.combinations(content, 2, axis=0, replacement=True))

    offsets = ak.index.Index64(np.array([0, 1, 292, 300], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(
        ak.combinations(cuda_content, 2, axis=0, replacement=False)
    ) == to_list(ak.combinations(content, 2, axis=0, replacement=False))
    assert to_list(
        ak.combinations(cuda_content, 2, axis=0, replacement=True)
    ) == to_list(ak.combinations(content, 2, axis=0, replacement=True))
    assert to_list(
        ak.combinations(cuda_content, 2, axis=-1, replacement=False)
    ) == to_list(ak.combinations(content, 2, axis=-1, replacement=False))
    assert to_list(
        ak.combinations(cuda_content, 2, axis=-1, replacement=True)
    ) == to_list(ak.combinations(content, 2, axis=-1, replacement=True))
    del cuda_content, cuda_depth1


def test_block_boundary_argcombinations():
    content = ak.contents.NumpyArray(np.arange(300))
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)

    assert to_list(
        ak.argcombinations(cuda_content, 2, axis=0, replacement=False)
    ) == to_list(ak.argcombinations(content, 2, axis=0, replacement=False))
    assert to_list(
        ak.argcombinations(cuda_content, 2, axis=0, replacement=True)
    ) == to_list(ak.argcombinations(content, 2, axis=0, replacement=True))

    offsets = ak.index.Index64(np.array([0, 1, 292, 300], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(
        ak.argcombinations(cuda_content, 2, axis=0, replacement=False)
    ) == to_list(ak.argcombinations(content, 2, axis=0, replacement=False))
    assert to_list(
        ak.argcombinations(cuda_content, 2, axis=0, replacement=True)
    ) == to_list(ak.argcombinations(content, 2, axis=0, replacement=True))

    del cuda_content, cuda_depth1
