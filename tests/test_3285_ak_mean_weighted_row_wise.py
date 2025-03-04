# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import math

import pytest

import awkward as ak


@pytest.mark.parametrize(
    "keepdims, expected_result",
    [
        pytest.param(False, ak.Array([2.25, 6.5])),
        pytest.param(True, ak.Array([[2.25], [6.5]])),
    ],
)
def test_keepdims(keepdims: bool, expected_result: ak.Array):
    data = ak.Array(
        [
            [1, 2, 3],
            [4, 7],
        ]
    )
    weight = ak.Array(
        [
            [1, 1, 2],
            [1, 5],
        ]
    )
    assert ak.all(
        ak.mean(data, weight=weight, axis=1, keepdims=keepdims) == expected_result
    )


@pytest.mark.parametrize(
    "mask_identity, expected_result",
    [
        pytest.param(False, ak.Array([1.5, math.nan, 8])),
        pytest.param(True, ak.Array([1.5, None, 8])),
    ],
)
def test_mask_identity(mask_identity: bool, expected_result: ak.Array):
    data = ak.Array(
        [
            [1, 2],
            [],
            [6, 9],
        ]
    )
    weight = ak.Array(
        [
            [1, 1],
            [],
            [1, 2],
        ]
    )
    result = ak.mean(data, weight=weight, axis=1, mask_identity=mask_identity)
    assert result[0] == expected_result[0]
    if mask_identity:
        assert result[1] is None
    else:
        assert math.isnan(result[1])  # NaN is not equal to itself per IEEE!
    assert result[2] == expected_result[2]
