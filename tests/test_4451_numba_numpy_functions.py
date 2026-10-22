# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")
ak.numba.register_and_check()


@pytest.mark.parametrize(
    "function",
    [
        np.sum,
        np.prod,
        np.min,
        np.max,
        np.mean,
        np.std,
        np.var,
        np.median,
        np.argmin,
        np.argmax,
        np.all,
        np.any,
        np.cumsum,
        np.cumprod,
        np.sort,
        np.argsort,
    ],
)
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64, np.bool_])
def test_flat(function, dtype):
    values = np.array([3, 0, 2, 1], dtype=dtype)
    array = ak.Array(values)

    @numba.njit
    def evaluate(a):
        return function(a)

    @numba.njit
    def explicit(a):
        return function(np.asarray(a))

    result = evaluate(array)
    np.testing.assert_allclose(result, explicit(array))
    np.testing.assert_allclose(result, function(values), rtol=1e-6)
    if isinstance(result, np.ndarray):
        assert result.dtype == explicit(array).dtype


def test_nested_iteration_and_slices():
    @numba.njit
    def evaluate(array):
        total = 0.0
        for row in array:
            total += np.sum(row[1:])
        return total

    assert evaluate(ak.Array([[10.0, 1.0, 2.0], [], [20.0, 3.0]])) == 6


def test_axis_and_dtype():
    @numba.njit
    def evaluate(array):
        return np.sum(array, axis=0, dtype=np.float64)

    values = np.array([1, 2, 3], dtype=np.int8)
    assert evaluate(ak.Array(values)) == np.sum(values, axis=0, dtype=np.float64)


@pytest.mark.parametrize("data", [[[1], [2]], [1, None], [{"x": 1}], ["x", "y"]])
def test_ineligible_layouts(data):
    @numba.njit
    def evaluate(array):
        return np.sum(array)

    with pytest.raises(numba.core.errors.TypingError):
        evaluate(ak.Array(data))


def test_numpy_inputs_unchanged():
    @numba.njit
    def evaluate(array):
        return np.sum(array), np.sort(array)

    total, ordered = evaluate(np.array([3, 1, 2]))
    assert total == 6
    np.testing.assert_array_equal(ordered, [1, 2, 3])


@pytest.mark.parametrize("function", [np.sum, np.prod, np.cumsum, np.sort])
def test_empty(function):
    @numba.njit
    def evaluate(array):
        return function(array)

    values = np.array([], dtype=np.float64)
    np.testing.assert_array_equal(evaluate(ak.Array(values)), function(values))


def test_sliced_input_and_complex():
    @numba.njit
    def evaluate(array):
        return np.sum(array), np.mean(array)

    array = ak.Array(np.array([99j, 1 + 2j, 3 + 4j, 99j]))[1:-1]
    assert evaluate(array) == (4 + 6j, 2 + 3j)


def test_record_field():
    @numba.njit
    def evaluate(array):
        return np.sum(array.x)

    assert evaluate(ak.Array([{"x": 1}, {"x": 2}])) == 3
