# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward._nplikes.array_module import _nplike_unique_has_equal_nan


def test_unique_flat_float16_extreme_values():
    # Widening must preserve subnormals and extremes. Repeated NaNs stay
    # separate when NumPy supports equal_nan=False; older NumPy merges them.
    # IEEE float16's smallest subnormal; older NumPy lacks smallest_subnormal.
    tiny = np.float16(2.0**-24)
    largest = np.finfo(np.float16).max
    data = np.array(
        [largest, tiny, -tiny, -largest, np.inf, -np.inf, 0, tiny, largest,
         np.nan, np.nan],
        dtype=np.float16,
    )  # fmt: skip
    out = ak._do.unique(ak.contents.NumpyArray(data), axis=None)
    nan_count = 2 if _nplike_unique_has_equal_nan(np) else 1
    expected = np.array(
        [-np.inf, -largest, -tiny, 0, tiny, largest, np.inf] + [np.nan] * nan_count,
        dtype=np.float16,
    )
    assert out.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(out.data, expected)
    # The operation must not alter the input buffer.
    assert data[0] == largest
    assert data[1] == tiny
