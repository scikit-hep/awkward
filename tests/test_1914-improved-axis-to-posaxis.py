# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_is_none():
    array = ak.Array(
        [
            None,
            [None],
            [{"x": None, "y": None}],
            [{"x": [None], "y": [None]}],
            [{"x": [1], "y": [[None]]}],
            [{"x": [2], "y": [[1, 2, 3]]}],
        ]
    )

    assert ak.is_none(array, axis=0).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert ak.is_none(array, axis=1).tolist() == [
        None,
        [True],
        [False],
        [False],
        [False],
        [False],
    ]
    assert ak.is_none(array, axis=2).tolist() == [
        None,
        [None],
        [{"x": None, "y": None}],
        [{"x": [True], "y": [True]}],
        [{"x": [False], "y": [False]}],
        [{"x": [False], "y": [False]}],
    ]

    with pytest.raises(np.AxisError):
        ak.is_none(array, axis=3)

    assert ak.is_none(array, axis=-1).tolist() == [
        None,
        [None],
        [{"x": None, "y": None}],
        [{"x": [True], "y": [None]}],
        [{"x": [False], "y": [[True]]}],
        [{"x": [False], "y": [[False, False, False]]}],
    ]
    assert ak.is_none(array, axis=-2).tolist() == [
        None,
        [None],
        [{"x": True, "y": None}],
        [{"x": False, "y": [True]}],
        [{"x": False, "y": [False]}],
        [{"x": False, "y": [False]}],
    ]

    with pytest.raises(np.AxisError):
        ak.is_none(array, axis=-3)


def test_singletons():
    array = ak.Array(
        [
            None,
            [None],
            [{"x": None, "y": None}],
            [{"x": [None], "y": [None]}],
            [{"x": [1], "y": [[None]]}],
            [{"x": [2], "y": [[1, 2, 3]]}],
        ]
    )

    assert ak.singletons(array, axis=0).tolist() == [
        [],
        [[None]],
        [[{"x": None, "y": None}]],
        [[{"x": [None], "y": [None]}]],
        [[{"x": [1], "y": [[None]]}]],
        [[{"x": [2], "y": [[1, 2, 3]]}]],
    ]

    assert ak.singletons(array, axis=1).tolist() == [
        None,
        [[]],
        [[{"x": None, "y": None}]],
        [[{"x": [None], "y": [None]}]],
        [[{"x": [1], "y": [[None]]}]],
        [[{"x": [2], "y": [[1, 2, 3]]}]],
    ]

    assert ak.singletons(array, axis=2).tolist() == [
        None,
        [None],
        [{"x": None, "y": None}],
        [{"x": [[]], "y": [[]]}],
        [{"x": [[1]], "y": [[[None]]]}],
        [{"x": [[2]], "y": [[[1, 2, 3]]]}],
    ]

    with pytest.raises(np.AxisError):
        ak.singletons(array, axis=3)

    assert ak.singletons(array, axis=-1).tolist() == [
        None,
        [None],
        [{"x": None, "y": None}],
        [{"x": [[]], "y": [None]}],
        [{"x": [[1]], "y": [[[]]]}],
        [{"x": [[2]], "y": [[[1], [2], [3]]]}],
    ]

    assert ak.singletons(array, axis=-2).tolist() == [
        None,
        [None],
        [{"x": [], "y": None}],
        [{"x": [[None]], "y": [[]]}],
        [{"x": [[1]], "y": [[[None]]]}],
        [{"x": [[2]], "y": [[[1, 2, 3]]]}],
    ]

    with pytest.raises(np.AxisError):
        ak.singletons(array, axis=-3)


def test_is_none_union():
    left = ak.Array([[[{"x": 1, "y": None}]]])
    right = ak.Array([[[{"a": 1, "b": None}]]])

    array = ak.concatenate([left, right], axis=2)

    assert ak.is_none(array, axis=-1).tolist() == [
        [[{"x": False, "y": True}, {"a": False, "b": True}]]
    ]
