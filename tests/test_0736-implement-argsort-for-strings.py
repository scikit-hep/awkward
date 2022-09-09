# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_but_first_fix_sort():
    assert ak._v2.operations.is_valid(
        ak._v2.operations.sort(ak._v2.Array(["one", "two", "three"]), axis=-1)
    )


def test_argsort():
    array = ak._v2.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    )
    assert ak._v2.operations.argsort(array, axis=-1).tolist() == [
        7,
        4,
        3,
        0,
        6,
        5,
        2,
        1,
    ]

    array = ak._v2.Array(
        [["twotwo", "two", "three"], ["four", "five"], [], ["six", "seven", "eight"]]
    )
    assert ak._v2.operations.argsort(array, axis=-1).tolist() == [
        [2, 1, 0],
        [1, 0],
        [],
        [2, 1, 0],
    ]

    array = ak._v2.Array(
        [
            [["twotwo", "two"], ["three"]],
            [["four", "five"]],
            [],
            [["six"], ["seven", "eight"]],
        ]
    )
    assert ak._v2.operations.argsort(array, axis=-1).tolist() == [
        [[1, 0], [0]],
        [[1, 0]],
        [],
        [[0], [1, 0]],
    ]


def test_sort():
    array = ak._v2.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    )
    assert ak._v2.operations.sort(array, axis=-1).tolist() == [
        "eight",
        "five",
        "four",
        "one",
        "seven",
        "six",
        "three",
        "two",
    ]

    array = ak._v2.Array(
        [["twotwo", "two", "three"], ["four", "five"], [], ["six", "seven", "eight"]]
    )
    assert ak._v2.operations.sort(array, axis=-1).tolist() == [
        ["three", "two", "twotwo"],
        ["five", "four"],
        [],
        ["eight", "seven", "six"],
    ]

    array = ak._v2.Array(
        [
            [["twotwo", "two"], ["three"]],
            [["four", "five"]],
            [],
            [["six"], ["seven", "eight"]],
        ]
    )
    assert ak._v2.operations.sort(array, axis=-1).tolist() == [
        [["two", "twotwo"], ["three"]],
        [["five", "four"]],
        [],
        [["six"], ["eight", "seven"]],
    ]
