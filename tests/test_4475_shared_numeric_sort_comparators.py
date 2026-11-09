# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import math

import numpy as np
import pytest

from awkward._backends.numpy import NumpyBackend

DTYPES = [
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset_dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("stable", [False, True])
def test_numeric_sort_and_argsort_order(dtype, offset_dtype, ascending, stable):
    kind = np.dtype(dtype).kind
    if kind == "f":
        sample = [np.nan, 0.0, -0.0, np.inf, -np.inf, 1.0, 1.0, np.nan, -1.0]
    elif kind == "b":
        sample = [True, False, True, False]
    else:
        info = np.iinfo(dtype)
        sample = [info.max, info.max - 1, info.min, info.min + 1, 0, 1, info.max]
    rows = [[], sample, sample[::-1], [1] * 100, []]
    values = np.asarray([x for row in rows for x in row], dtype=dtype)
    offsets = np.asarray(
        [0, *np.cumsum([len(row) for row in rows])], dtype=offset_dtype
    )
    result = np.empty_like(values)
    indices = np.empty(len(values), dtype=np.int64)
    backend = NumpyBackend.instance()
    sort = backend["awkward_sort", dtype, dtype, np.int64]
    argsort = backend["awkward_argsort", np.int64, dtype, offset_dtype]
    assert (
        sort(
            result,
            values,
            len(values),
            offsets.astype(np.int64),
            len(offsets),
            ascending,
            stable,
        ).str
        is None
    )
    assert (
        argsort(
            indices, values, len(values), offsets, len(offsets), ascending, stable
        ).str
        is None
    )
    position = 0
    for row in rows:
        # NaNs lead in either direction; compare Python integers without rounding.
        def key(i, row=row):
            value = row[i]
            if kind == "f" and math.isnan(value):
                return (0, 0)
            return (1, value if ascending else -value)

        expected_indices = sorted(range(len(row)), key=key)
        expected_values = np.asarray([row[i] for i in expected_indices], dtype=dtype)
        stop = position + len(row)
        actual_indices = indices[position:stop]
        np.testing.assert_array_equal(np.sort(actual_indices), np.arange(len(row)))
        np.testing.assert_array_equal(result[position:stop], expected_values)
        np.testing.assert_array_equal(
            np.asarray(row, dtype=dtype)[actual_indices], expected_values
        )
        if stable:
            np.testing.assert_array_equal(actual_indices, expected_indices)
            if kind == "f":
                unsigned = np.dtype(f"u{np.dtype(dtype).itemsize}")
                np.testing.assert_array_equal(
                    result[position:stop].view(unsigned), expected_values.view(unsigned)
                )
        position = stop
