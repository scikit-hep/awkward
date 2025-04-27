# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

jax = pytest.importorskip("jax")
ak.jax.register_and_check()

# Define all reducers to test
REDUCERS = [
    (ak.argmin, {}),
    (ak.argmax, {}),
    (ak.min, {}),
    (ak.max, {}),
    (ak.sum, {}),
    (ak.prod, {"mask_identity": True}),  # mask_identity for prod to handle empty arrays
    (ak.any, {}),
    (ak.all, {}),
    (ak.count, {}),
    (ak.count_nonzero, {}),
]

# Define test arrays (single jagged)
SINGLE_JAGGED = [
    # Normal array
    [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    # Array with first empty
    [[], [1, 2], [3, 4, 5]],
    # Array with middle empty
    [[1, 2], [], [3, 4, 5]],
    # Array with last empty
    [[1, 2], [3, 4, 5], []],
    # Array with multiple empty elements
    [[], [1, 2], [], [3, 4], []],
    # Array with negative numbers
    [[-1, -2], [-3], [4, 5, -6]],
    # Array with zeros
    [[0, 0], [1, 0], [2, 3, 4]],
]

# Define test arrays (double jagged)
DOUBLE_JAGGED = [
    # Normal double jagged array
    [[[1, 2], [3]], [[4, 5, 6]], [[7], [8, 9]]],
    # Double jagged with empty at first level
    [[], [[1, 2], [3, 4]], [[5, 6]]],
    # Double jagged with empty at second level
    [[[1, 2], []], [[3, 4], [5]], [[6]]],
    # Double jagged with various empty elements
    [[[]], [[], [1, 2]], [[], [], [3, 4]]],
    # Double jagged with negative numbers
    [[[-1, -2], [-3]], [[4, 5, -6]], [[7], [-8, 9]]],
    # Double jagged with zeros
    [[[0, 0], [1]], [[2, 3]], [[4], [5, 6]]],
]

# Define axes to test
AXES = [1, None]  # axis=1 for first dimension, None for flattened reduction
DOUBLE_JAGGED_AXES = [1, 2, None]  # axis=1 and axis=2 for double jagged

RTOL = 1e-5  # Relative tolerance for floating point comparison
ATOL = 1e-8  # Absolute tolerance for floating point comparison


def compare_results(cpu_list, jax_list):
    """Compare results with tolerance for numeric values."""
    if isinstance(cpu_list, (int, float)) and isinstance(jax_list, (int, float)):
        # Direct numeric comparison with tolerance
        np.testing.assert_allclose(cpu_list, jax_list, rtol=RTOL, atol=ATOL)
    elif isinstance(cpu_list, list) and isinstance(jax_list, list):
        # Lists should have the same length
        assert len(cpu_list) == len(jax_list), (
            f"Lists have different lengths: {len(cpu_list)} vs {len(jax_list)}"
        )

        # Compare each element
        for cpu_item, jax_item in zip(cpu_list, jax_list):
            compare_results(cpu_item, jax_item)
    else:
        # For non-numeric types, use exact equality
        assert cpu_list == jax_list


@pytest.mark.parametrize("reducer,kwargs", REDUCERS)
@pytest.mark.parametrize("arr", SINGLE_JAGGED)
@pytest.mark.parametrize("axis", AXES)
def test_single_jagged_arrays(reducer, kwargs, arr, axis):
    """Test reducers on single jagged arrays with different axes."""

    # Create arrays with different backends
    cpu_array = ak.Array(arr, backend="cpu")
    jax_array = ak.Array(arr, backend="jax")

    # Apply reducers to each backend's array
    cpu_result = reducer(cpu_array, axis=axis, **kwargs)
    jax_result = reducer(jax_array, axis=axis, **kwargs)

    # Convert to lists for comparison
    cpu_list = ak.to_list(cpu_result)
    jax_list = ak.to_list(jax_result)

    # Handle case where axis=None might result in different structures
    if axis is None:
        # If one result is a scalar and the other is a list with one element
        if (
            not isinstance(cpu_list, list)
            and isinstance(jax_list, list)
            and len(jax_list) == 1
        ):
            jax_list = jax_list[0]
        elif (
            isinstance(cpu_list, list)
            and not isinstance(jax_list, list)
            and len(cpu_list) == 1
        ):
            cpu_list = cpu_list[0]

    # Compare with tolerance for numeric values
    compare_results(cpu_list, jax_list)


@pytest.mark.parametrize("reducer,kwargs", REDUCERS)
@pytest.mark.parametrize("arr", DOUBLE_JAGGED)
@pytest.mark.parametrize("axis", DOUBLE_JAGGED_AXES)
def test_double_jagged_arrays(reducer, kwargs, arr, axis):
    """Test reducers on double jagged arrays with different axes."""

    # Create arrays with different backends
    cpu_array = ak.Array(arr, backend="cpu")
    jax_array = ak.Array(arr, backend="jax")

    # Apply reducers to each backend's array
    cpu_result = reducer(cpu_array, axis=axis, **kwargs)
    jax_result = reducer(jax_array, axis=axis, **kwargs)

    # Convert to lists for comparison
    cpu_list = ak.to_list(cpu_result)
    jax_list = ak.to_list(jax_result)

    # Handle case where axis=None might result in different structures
    if axis is None:
        # If one result is a scalar and the other is a list with one element
        if (
            not isinstance(cpu_list, list)
            and isinstance(jax_list, list)
            and len(jax_list) == 1
        ):
            jax_list = jax_list[0]
        elif (
            isinstance(cpu_list, list)
            and not isinstance(jax_list, list)
            and len(cpu_list) == 1
        ):
            cpu_list = cpu_list[0]

    # Compare with tolerance for numeric values
    compare_results(cpu_list, jax_list)


# Additional edge cases
@pytest.mark.parametrize("reducer,kwargs", REDUCERS)
def test_all_empty_arrays(reducer, kwargs):
    """Test with arrays that are entirely empty."""

    all_empty_data = [[], [], []]
    cpu_array = ak.Array(all_empty_data, backend="cpu")
    jax_array = ak.Array(all_empty_data, backend="jax")

    cpu_result = reducer(cpu_array, axis=1, **kwargs)
    jax_result = reducer(jax_array, axis=1, **kwargs)

    # Convert to lists for comparison
    cpu_list = ak.to_list(cpu_result)
    jax_list = ak.to_list(jax_result)

    # Handle case where one might be a scalar and the other a list
    if (
        not isinstance(cpu_list, list)
        and isinstance(jax_list, list)
        and len(jax_list) == 1
    ):
        jax_list = jax_list[0]
    elif (
        isinstance(cpu_list, list)
        and not isinstance(jax_list, list)
        and len(cpu_list) == 1
    ):
        cpu_list = cpu_list[0]

    # Compare with tolerance for numeric values
    compare_results(cpu_list, jax_list)


# Test with boolean values
@pytest.mark.parametrize("reducer,kwargs", REDUCERS)
def test_boolean_arrays(reducer, kwargs):
    """Test with boolean arrays."""

    bool_data = [[True, False], [], [True, True, False], [False]]
    cpu_array = ak.Array(bool_data, backend="cpu")
    jax_array = ak.Array(bool_data, backend="jax")

    cpu_result = reducer(cpu_array, axis=1, **kwargs)
    jax_result = reducer(jax_array, axis=1, **kwargs)

    # Convert to lists for comparison
    cpu_list = ak.to_list(cpu_result)
    jax_list = ak.to_list(jax_result)

    # Handle case where one might be a scalar and the other a list
    if (
        not isinstance(cpu_list, list)
        and isinstance(jax_list, list)
        and len(jax_list) == 1
    ):
        jax_list = jax_list[0]
    elif (
        isinstance(cpu_list, list)
        and not isinstance(jax_list, list)
        and len(cpu_list) == 1
    ):
        cpu_list = cpu_list[0]

    # Compare with tolerance for numeric values
    compare_results(cpu_list, jax_list)


# Test with None values
@pytest.mark.parametrize("reducer,kwargs", REDUCERS)
def test_none_arrays(reducer, kwargs):
    """Test with arrays containing None values."""

    none_data = [[None, 1], [2, None], [None, None], [3, 4]]
    cpu_array = ak.Array(none_data, backend="cpu")
    jax_array = ak.Array(none_data, backend="jax")

    cpu_result = reducer(cpu_array, axis=1, **kwargs)
    jax_result = reducer(jax_array, axis=1, **kwargs)

    # Convert to lists for comparison
    cpu_list = ak.to_list(cpu_result)
    jax_list = ak.to_list(jax_result)

    # Handle case where one might be a scalar and the other a list
    if (
        not isinstance(cpu_list, list)
        and isinstance(jax_list, list)
        and len(jax_list) == 1
    ):
        jax_list = jax_list[0]
    elif (
        isinstance(cpu_list, list)
        and not isinstance(jax_list, list)
        and len(cpu_list) == 1
    ):
        cpu_list = cpu_list[0]

    # Compare with tolerance for numeric values
    compare_results(cpu_list, jax_list)


# test with NaN values
@pytest.mark.skip(
    reason="(arg)min/max and any do not work with NaNs in the jax backend"
)
@pytest.mark.parametrize("reducer,kwargs", REDUCERS)
def test_nan_arrays(reducer, kwargs):
    """Test with arrays containing NaN values."""

    nan_data = [[np.nan, 1], [2, np.nan], [np.nan, np.nan], [3, 4]]
    cpu_array = ak.Array(nan_data, backend="cpu")
    jax_array = ak.Array(nan_data, backend="jax")

    cpu_result = reducer(cpu_array, axis=1, **kwargs)
    jax_result = reducer(jax_array, axis=1, **kwargs)

    # Convert to lists for comparison
    cpu_list = ak.to_list(cpu_result)
    jax_list = ak.to_list(jax_result)

    # Handle case where one might be a scalar and the other a list
    if (
        not isinstance(cpu_list, list)
        and isinstance(jax_list, list)
        and len(jax_list) == 1
    ):
        jax_list = jax_list[0]
    elif (
        isinstance(cpu_list, list)
        and not isinstance(jax_list, list)
        and len(cpu_list) == 1
    ):
        cpu_list = cpu_list[0]

    # Compare with tolerance for numeric values
    compare_results(cpu_list, jax_list)
