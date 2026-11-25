# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


def test_softmax_is_stable_for_large_finite_values():
    array = ak.Array([[1000.0, 1001.0], [-1000.0, -1001.0]])

    result = ak.softmax(array, axis=-1)

    np.testing.assert_allclose(
        ak.to_numpy(result),
        [[0.26894142, 0.73105858], [0.73105858, 0.26894142]],
    )

    with_empty_list = ak.Array([[1000.0, 1001.0], []])
    assert ak.softmax(with_empty_list, axis=-1, mask_identity=True).to_list()[1] is None


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint64, np.int64, np.bool_, np.float32, np.float64]
)
@pytest.mark.parametrize("keepdims", [False, True])
def test_softmax_regular_inputs(dtype, keepdims):
    values = np.array([[1, 2], [0, 1]], dtype=dtype)
    array = ak.Array(values)

    floating = values.astype(np.float64)
    shifted = floating - floating.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    expected = exponential / exponential.sum(axis=-1, keepdims=True)

    result = ak.to_numpy(ak.softmax(array, keepdims=keepdims))
    np.testing.assert_allclose(result, expected)
