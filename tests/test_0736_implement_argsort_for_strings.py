# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import numpy as np  # noqa: F401
import pytest

import awkward as ak


def test_but_first_fix_sort():
    assert ak.operations.is_valid(
        ak.operations.sort(ak.Array(["one", "two", "three"]), axis=-1)
    )


def test_argsort():
    array = ak.Array(["one", "two", "three", "four", "five", "six", "seven", "eight"])
    assert ak.operations.argsort(array, axis=-1).to_list() == [
        7,
        4,
        3,
        0,
        6,
        5,
        2,
        1,
    ]

    array = ak.Array(
        [["twotwo", "two", "three"], ["four", "five"], [], ["six", "seven", "eight"]]
    )
    assert ak.operations.argsort(array, axis=-1).to_list() == [
        [2, 1, 0],
        [1, 0],
        [],
        [2, 1, 0],
    ]

    array = ak.Array(
        [
            [["twotwo", "two"], ["three"]],
            [["four", "five"]],
            [],
            [["six"], ["seven", "eight"]],
        ]
    )
    assert ak.operations.argsort(array, axis=-1).to_list() == [
        [[1, 0], [0]],
        [[1, 0]],
        [],
        [[0], [1, 0]],
    ]


def test_sort():
    array = ak.Array(["one", "two", "three", "four", "five", "six", "seven", "eight"])
    assert ak.operations.sort(array, axis=-1).to_list() == [
        "eight",
        "five",
        "four",
        "one",
        "seven",
        "six",
        "three",
        "two",
    ]

    array = ak.Array(
        [["twotwo", "two", "three"], ["four", "five"], [], ["six", "seven", "eight"]]
    )
    assert ak.operations.sort(array, axis=-1).to_list() == [
        ["three", "two", "twotwo"],
        ["five", "four"],
        [],
        ["eight", "seven", "six"],
    ]

    array = ak.Array(
        [
            [["twotwo", "two"], ["three"]],
            [["four", "five"]],
            [],
            [["six"], ["seven", "eight"]],
        ]
    )
    assert ak.operations.sort(array, axis=-1).to_list() == [
        [["two", "twotwo"], ["three"]],
        [["five", "four"]],
        [],
        [["six"], ["eight", "seven"]],
    ]


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("kwargs", [{}, {"stable": True}])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("keys", [["a", "a"], ["b", "a", "aa", "b", "", "a", "aa", ""]])
def test_argsort_strings_stable(keys, nested, kwargs, ascending):
    expected = sorted(range(len(keys)), key=keys.__getitem__, reverse=not ascending)
    if nested:
        array = ak.Array([keys, [], keys])
        expected = [expected, [], expected]
    else:
        array = ak.Array(keys)

    assert ak.argsort(array, ascending=ascending, **kwargs).to_list() == expected
