from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    try:
        cp.cuda.Device().synchronize()  # wait for all kernels
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("GPU error during sync:", e)
    cp._default_memory_pool.free_all_blocks()


def test_combinations_ListOffsetArray_n3():
    array = ak.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )
    array = ak.to_backend(array, "cuda")

    # combinations without replacement
    expected_no_replacement = [
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
    assert (
        to_list(ak.combinations(array, 3, replacement=False)) == expected_no_replacement
    )

    # combinations without replacement, named fields
    expected_fields = [
        [
            {"x": 0.0, "y": 1.1, "z": 2.2},
            {"x": 0.0, "y": 1.1, "z": 3.3},
            {"x": 0.0, "y": 2.2, "z": 3.3},
            {"x": 1.1, "y": 2.2, "z": 3.3},
        ],
        [],
        [{"x": 4.4, "y": 5.5, "z": 6.6}],
        [],
        [
            {"x": 8.8, "y": 9.9, "z": 10.0},
            {"x": 8.8, "y": 9.9, "z": 11.1},
            {"x": 8.8, "y": 9.9, "z": 12.2},
            {"x": 8.8, "y": 10.0, "z": 11.1},
            {"x": 8.8, "y": 10.0, "z": 12.2},
            {"x": 8.8, "y": 11.1, "z": 12.2},
            {"x": 9.9, "y": 10.0, "z": 11.1},
            {"x": 9.9, "y": 10.0, "z": 12.2},
            {"x": 9.9, "y": 11.1, "z": 12.2},
            {"x": 10.0, "y": 11.1, "z": 12.2},
        ],
    ]
    assert (
        to_list(ak.combinations(array, 3, replacement=False, fields=["x", "y", "z"]))
        == expected_fields
    )

    # combinations with parameters
    tmp = ak.combinations(
        array, 3, replacement=False, parameters={"some": "param"}
    ).layout
    assert tmp.content.parameters["some"] == "param"

    # combinations with replacement
    expected_with_replacement = [
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
    assert (
        to_list(ak.combinations(array, 3, replacement=True))
        == expected_with_replacement
    )

    del array


def test_combinations_ListOffsetArray_n3_2():
    array = ak.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )
    array = ak.to_backend(array, "cuda")

    # combinations of 3 without replacement
    assert to_list(ak.combinations(array, 3, replacement=False)) == [
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

    # combinations with fields
    assert to_list(
        ak.combinations(array, 3, replacement=False, fields=["x", "y", "z"])
    ) == [
        [
            {"x": 0.0, "y": 1.1, "z": 2.2},
            {"x": 0.0, "y": 1.1, "z": 3.3},
            {"x": 0.0, "y": 2.2, "z": 3.3},
            {"x": 1.1, "y": 2.2, "z": 3.3},
        ],
        [],
        [{"x": 4.4, "y": 5.5, "z": 6.6}],
        [],
        [
            {"x": 8.8, "y": 9.9, "z": 10.0},
            {"x": 8.8, "y": 9.9, "z": 11.1},
            {"x": 8.8, "y": 9.9, "z": 12.2},
            {"x": 8.8, "y": 10.0, "z": 11.1},
            {"x": 8.8, "y": 10.0, "z": 12.2},
            {"x": 8.8, "y": 11.1, "z": 12.2},
            {"x": 9.9, "y": 10.0, "z": 11.1},
            {"x": 9.9, "y": 10.0, "z": 12.2},
            {"x": 9.9, "y": 11.1, "z": 12.2},
            {"x": 10.0, "y": 11.1, "z": 12.2},
        ],
    ]

    # with parameters
    tmp = ak.combinations(
        array, 3, replacement=False, parameters={"some": "param"}
    ).layout
    assert tmp.content.parameters["some"] == "param"

    # combinations with replacement
    assert to_list(ak.combinations(array, 3, replacement=True)) == [
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

    del array


@pytest.mark.skip("RegularArray combinations for n>2 is not implemented")
def test_combinations_RegularArray_n3():
    array = ak.Array(np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]]))
    array = ak.to_backend(array, "cuda")

    # combinations of length 3, without replacement
    print(to_list(ak.combinations(array, 3, replacement=False)))
    assert to_list(ak.combinations(array, 3, replacement=False)) == [
        [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
        [(4.4, 5.5, 6.6), (4.4, 5.5, 7.7), (4.4, 6.6, 7.7), (5.5, 6.6, 7.7)],
    ]

    # combinations of length 3, without replacement, with fields
    assert to_list(
        ak.combinations(array, 3, replacement=False, fields=["x", "y", "z"])
    ) == [
        [
            {"x": 0.0, "y": 1.1, "z": 2.2},
            {"x": 0.0, "y": 1.1, "z": 3.3},
            {"x": 0.0, "y": 2.2, "z": 3.3},
            {"x": 1.1, "y": 2.2, "z": 3.3},
        ],
        [
            {"x": 4.4, "y": 5.5, "z": 6.6},
            {"x": 4.4, "y": 5.5, "z": 7.7},
            {"x": 4.4, "y": 6.6, "z": 7.7},
            {"x": 5.5, "y": 6.6, "z": 7.7},
        ],
    ]

    # test parameters are propagated
    tmp = ak.combinations(
        array, 3, replacement=False, parameters={"some": "param"}
    ).layout
    assert tmp.content.parameters["some"] == "param"

    # combinations of length 3, with replacement
    assert to_list(ak.combinations(array, 3, replacement=True)) == [
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
    del array


@pytest.mark.skip("RegularArray combinations for n>2 is not implemented")
def test_combinations_axis0_n3_1():
    array = ak.Array([0.0, 1.1, 2.2, 3.3])
    array = ak.to_backend(array, "cuda")

    # combinations of length 3, without replacement, axis=0
    assert to_list(ak.combinations(array, 3, replacement=False, axis=0)) == [
        (0.0, 1.1, 2.2),
        (0.0, 1.1, 3.3),
        (0.0, 2.2, 3.3),
        (1.1, 2.2, 3.3),
    ]

    # combinations of length 3, without replacement, axis=0, with fields
    assert to_list(
        ak.combinations(array, 3, replacement=False, axis=0, fields=["x", "y", "z"])
    ) == [
        {"x": 0.0, "y": 1.1, "z": 2.2},
        {"x": 0.0, "y": 1.1, "z": 3.3},
        {"x": 0.0, "y": 2.2, "z": 3.3},
        {"x": 1.1, "y": 2.2, "z": 3.3},
    ]

    # parameters propagation
    assert (
        ak.combinations(
            array, 3, replacement=False, axis=0, parameters={"some": "param"}
        ).layout.parameters["some"]
        == "param"
    )
    del array


def test_combinations_IndexedArray_n3():
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

    # combinations of length 3, without replacement
    assert to_list(ak.combinations(array, 3, replacement=False)) == [
        [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
        [],
        [(4.4, 5.5, 6.6)],
        None,
        [],
        None,
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
    del array


def test_combinations_axis2_n3():
    array = ak.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    )
    array = ak.to_backend(array, "cuda")

    # combinations along axis=1, length 3
    assert to_list(ak.combinations(array, 3, axis=1, replacement=False)) == [
        [
            ([0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]),
        ],
        [],
        [],
    ]

    # combinations along axis=2, length 3
    assert to_list(ak.combinations(array, 3, axis=2, replacement=False)) == [
        [
            [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
            [],
            [(4.4, 5.5, 6.6)],
        ],
        [],
        [
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
        ],
    ]
    del array


def test_combinations_n3():
    array = ak.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    )
    array = ak.to_backend(array, "cuda")

    assert to_list(ak.operations.argcombinations(array, 3, replacement=False)) == [
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        [],
        [(0, 1, 2)],
        [],
        [
            (0, 1, 2),
            (0, 1, 3),
            (0, 1, 4),
            (0, 2, 3),
            (0, 2, 4),
            (0, 3, 4),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 4),
            (2, 3, 4),
        ],
    ]
    del array


def test_combinations_ListOffsetArray_n3_3():
    array = ak.Array(
        [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6], [7.7], [8.8, 9.9, 10.0, 11.1, 12.2]]
    ).layout
    array = ak.to_backend(array, "cuda", highlevel=False)

    # Basic combinations without replacement
    assert to_list(ak._do.combinations(array, 3, replacement=False)) == [
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
    # Typetracer consistency
    assert (
        ak._do.combinations(array.to_typetracer(), 3, replacement=False).form
        == ak._do.combinations(array, 3, replacement=False).form
    )

    # Named fields
    assert to_list(
        ak._do.combinations(array, 3, replacement=False, fields=["x", "y", "z"])
    ) == [
        [
            {"x": 0.0, "y": 1.1, "z": 2.2},
            {"x": 0.0, "y": 1.1, "z": 3.3},
            {"x": 0.0, "y": 2.2, "z": 3.3},
            {"x": 1.1, "y": 2.2, "z": 3.3},
        ],
        [],
        [{"x": 4.4, "y": 5.5, "z": 6.6}],
        [],
        [
            {"x": 8.8, "y": 9.9, "z": 10.0},
            {"x": 8.8, "y": 9.9, "z": 11.1},
            {"x": 8.8, "y": 9.9, "z": 12.2},
            {"x": 8.8, "y": 10.0, "z": 11.1},
            {"x": 8.8, "y": 10.0, "z": 12.2},
            {"x": 8.8, "y": 11.1, "z": 12.2},
            {"x": 9.9, "y": 10.0, "z": 11.1},
            {"x": 9.9, "y": 10.0, "z": 12.2},
            {"x": 9.9, "y": 11.1, "z": 12.2},
            {"x": 10.0, "y": 11.1, "z": 12.2},
        ],
    ]
    assert (
        ak._do.combinations(
            array.to_typetracer(), 3, replacement=False, fields=["x", "y", "z"]
        ).form
        == ak._do.combinations(array, 3, replacement=False, fields=["x", "y", "z"]).form
    )

    # Replacement combinations
    assert to_list(ak._do.combinations(array, 3, replacement=True)) == [
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
    assert (
        ak._do.combinations(array.to_typetracer(), 2, replacement=True).form
        == ak._do.combinations(array, 2, replacement=True).form
    )
    del array


@pytest.mark.skip("RegularArray combinations for n>2 is not implemented")
def test_combinations_RegularArray_n3_2():
    array = ak.Array(np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7]])).layout
    array = ak.to_backend(array, "cuda", highlevel=False)

    # combinations without replacement
    assert to_list(ak._do.combinations(array, 3, replacement=False)) == [
        [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
        [(4.4, 5.5, 6.6), (4.4, 5.5, 7.7), (4.4, 6.6, 7.7), (5.5, 6.6, 7.7)],
    ]
    assert (
        ak._do.combinations(array.to_typetracer(), 3, replacement=False).form
        == ak._do.combinations(array, 3, replacement=False).form
    )

    # combinations with field names
    assert to_list(
        ak._do.combinations(array, 3, replacement=False, fields=["x", "y", "z"])
    ) == [
        [
            {"x": 0.0, "y": 1.1, "z": 2.2},
            {"x": 0.0, "y": 1.1, "z": 3.3},
            {"x": 0.0, "y": 2.2, "z": 3.3},
            {"x": 1.1, "y": 2.2, "z": 3.3},
        ],
        [
            {"x": 4.4, "y": 5.5, "z": 6.6},
            {"x": 4.4, "y": 5.5, "z": 7.7},
            {"x": 4.4, "y": 6.6, "z": 7.7},
            {"x": 5.5, "y": 6.6, "z": 7.7},
        ],
    ]
    assert (
        ak._do.combinations(
            array.to_typetracer(), 3, replacement=False, fields=["x", "y", "z"]
        ).form
        == ak._do.combinations(array, 3, replacement=False, fields=["x", "y", "z"]).form
    )

    # combinations with parameters
    assert (
        ak._do.combinations(
            array, 3, replacement=False, parameters={"some": "param"}
        ).content.parameters["some"]
        == "param"
    )
    assert (
        ak._do.combinations(
            array.to_typetracer(), 3, replacement=False, parameters={"some": "param"}
        ).form
        == ak._do.combinations(
            array, 3, replacement=False, parameters={"some": "param"}
        ).form
    )

    # combinations with replacement
    assert to_list(ak._do.combinations(array, 3, replacement=True)) == [
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
    assert (
        ak._do.combinations(array.to_typetracer(), 3, replacement=True).form
        == ak._do.combinations(array, 3, replacement=True).form
    )
    del array


@pytest.mark.skip("RegularArray combinations for n>2 is not implemented")
def test_combinations_axis0_n3_2():
    array = ak.Array([0.0, 1.1, 2.2, 3.3]).layout
    array = ak.to_backend(array, "cuda", highlevel=False)

    # Combinations of length 3 without replacement
    assert to_list(ak._do.combinations(array, 3, replacement=False, axis=0)) == [
        (0.0, 1.1, 2.2),
        (0.0, 1.1, 3.3),
        (0.0, 2.2, 3.3),
        (1.1, 2.2, 3.3),
    ]
    assert (
        ak._do.combinations(array.to_typetracer(), 3, replacement=False, axis=0).form
        == ak._do.combinations(array, 3, replacement=False, axis=0).form
    )

    # Combinations with field names
    assert to_list(
        ak._do.combinations(array, 3, replacement=False, axis=0, fields=["x", "y", "z"])
    ) == [
        {"x": 0.0, "y": 1.1, "z": 2.2},
        {"x": 0.0, "y": 1.1, "z": 3.3},
        {"x": 0.0, "y": 2.2, "z": 3.3},
        {"x": 1.1, "y": 2.2, "z": 3.3},
    ]
    assert (
        ak._do.combinations(
            array.to_typetracer(),
            3,
            replacement=False,
            axis=0,
            fields=["x", "y", "z"],
        ).form
        == ak._do.combinations(
            array, 3, replacement=False, axis=0, fields=["x", "y", "z"]
        ).form
    )

    # Combinations with parameters
    assert (
        ak._do.combinations(
            array, 3, replacement=False, axis=0, parameters={"some": "param"}
        ).parameters["some"]
        == "param"
    )
    assert (
        ak._do.combinations(
            array.to_typetracer(),
            3,
            replacement=False,
            axis=0,
            parameters={"some": "param"},
        ).form
        == ak._do.combinations(
            array, 3, replacement=False, axis=0, parameters={"some": "param"}
        ).form
    )

    del array


def test_combinations_IndexedArray_n3_2():
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
    ).layout
    array = ak.to_backend(array, "cuda", highlevel=False)

    # Combinations of length 3 without replacement
    assert to_list(ak._do.combinations(array, 3, replacement=False)) == [
        [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
        [],
        [(4.4, 5.5, 6.6)],
        None,
        [],
        None,
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
    assert (
        ak._do.combinations(array.to_typetracer(), 3, replacement=False).form
        == ak._do.combinations(array, 3, replacement=False).form
    )
    del array


def test_1074_combinations_n3_axis2():
    array = ak.Array(
        [
            [[0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6]],
            [],
            [[7.7], [8.8, 9.9, 10.0, 11.1, 12.2]],
        ]
    ).layout
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(array, 3, axis=1, replacement=False)) == [
        [([0.0, 1.1, 2.2, 3.3], [], [4.4, 5.5, 6.6])],
        [],
        [],
    ]
    assert (
        ak._do.combinations(array.to_typetracer(), 3, axis=1, replacement=False).form
        == ak._do.combinations(array, 3, axis=1, replacement=False).form
    )

    assert to_list(ak._do.combinations(array, 3, axis=2, replacement=False)) == [
        [
            [(0.0, 1.1, 2.2), (0.0, 1.1, 3.3), (0.0, 2.2, 3.3), (1.1, 2.2, 3.3)],
            [],
            [(4.4, 5.5, 6.6)],
        ],
        [],
        [
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
        ],
    ]
    assert (
        ak._do.combinations(array.to_typetracer(), 3, axis=2, replacement=False).form
        == ak._do.combinations(array, 3, axis=2, replacement=False).form
    )
    del array


def test_1074_combinations_EmptyArray():
    array = ak.contents.emptyarray.EmptyArray()
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak._do.combinations(array, 3, axis=0)) == []
    assert (
        ak._do.combinations(array.to_typetracer(), 3, axis=0).form
        == ak._do.combinations(array, 3, axis=0).form
    )
    del array
