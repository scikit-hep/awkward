# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_nan():
    array = ak.Array([1, 2, np.nan, 3, 0, np.nan])

    assert ak.operations.argsort(array).to_list() == [
        2,
        5,
        4,
        0,
        1,
        3,
    ]
    # Note, `nan` comparison with `nan` returns False
    assert str(ak.operations.sort(array).to_list()) == "[nan, nan, 0.0, 1.0, 2.0, 3.0]"


def test_bool():
    array = ak.Array([True, False, False, True, True, True])

    assert ak.operations.argsort(array).to_list() == [1, 2, 0, 3, 4, 5]
    assert ak.operations.sort(array).to_list() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_argsort():
    array = ak.Array([1, 2, None, 3, 0, None])
    assert ak.operations.argsort(array).to_list() == [4, 0, 1, 3, 2, 5]
    assert array[ak.operations.argsort(array)].to_list() == [
        0,
        1,
        2,
        3,
        None,
        None,
    ]


def test_argsort_2d():
    array = ak.Array([[1, 2, None, 3, 0, None], [1, 2, None, 3, 0, None]])
    assert ak.operations.argsort(array).to_list() == [
        [4, 0, 1, 3, 2, 5],
        [4, 0, 1, 3, 2, 5],
    ]
    assert array[ak.operations.argsort(array)].to_list() == [
        [
            0,
            1,
            2,
            3,
            None,
            None,
        ],
        [
            0,
            1,
            2,
            3,
            None,
            None,
        ],
    ]
