# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest

import awkward as ak


def test_tuple_tuple():
    tuple_1 = ak.Array(
        [
            [
                ([1, 5, 1], [2, 5, 1]),
                ([3, 5, 1], [4, 5, 1]),
            ]
        ]
    )
    tuple_2 = ak.Array(
        [
            [
                ([1, 5, 1], [9, 10, 11]),
                ([6, 7, 8], [4, 5, 1]),
            ]
        ]
    )

    result = ak.concatenate([tuple_1, tuple_2], axis=-1)
    assert ak.almost_equal(
        result,
        [
            [
                ([1, 5, 1, 1, 5, 1], [2, 5, 1, 9, 10, 11]),
                ([3, 5, 1, 6, 7, 8], [4, 5, 1, 4, 5, 1]),
            ]
        ],
    )


def test_record_tuple():
    record_1 = ak.Array(
        [
            [
                {"0": [1, 5, 1], "1": [2, 5, 1]},
                {"0": [3, 5, 1], "1": [4, 5, 1]},
            ]
        ]
    )
    tuple_2 = ak.Array(
        [
            [
                ([1, 5, 1], [9, 10, 11]),
                ([6, 7, 8], [4, 5, 1]),
            ]
        ]
    )

    with pytest.raises(TypeError):
        ak.concatenate([record_1, tuple_2], axis=-1)


def test_record_record():
    record_1 = ak.Array(
        [
            [
                {"0": [1, 5, 1], "1": [2, 5, 1]},
                {"0": [3, 5, 1], "1": [4, 5, 1]},
            ]
        ]
    )
    record_2 = ak.Array(
        [
            [
                {"0": [1, 5, 1], "1": [9, 10, 11]},
                {"0": [6, 7, 8], "1": [4, 5, 1]},
            ]
        ]
    )

    result = ak.concatenate([record_1, record_2], axis=-1)
    assert ak.almost_equal(
        result,
        [
            [
                {"0": [1, 5, 1, 1, 5, 1], "1": [2, 5, 1, 9, 10, 11]},
                {"0": [3, 5, 1, 6, 7, 8], "1": [4, 5, 1, 4, 5, 1]},
            ]
        ],
    )
