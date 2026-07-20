# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_sort_int64_above_2_53_ascending():
    """ak.sort on int64 values > 2^53 must match numpy, ascending."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.sort(arr, ascending=True)
    expected = np.sort(np.array(vals, dtype=np.int64)).tolist()
    assert result.tolist()[0] == expected


def test_argsort_int64_above_2_53_ascending():
    """ak.argsort on int64 values > 2^53 must match numpy, ascending."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.argsort(arr, ascending=True)
    expected = np.argsort(np.array(vals, dtype=np.int64), kind="stable").tolist()
    assert result.tolist()[0] == expected


def test_sort_uint64_above_2_53_ascending():
    """ak.sort on uint64 values > 2^53 must match numpy, ascending."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array(np.array([vals], dtype=np.uint64))
    result = ak.sort(arr, ascending=True)
    expected = np.sort(np.array(vals, dtype=np.uint64)).tolist()
    assert result.tolist()[0] == expected


def test_sort_float_nan_ascending():
    """NaN handling for float sort must be unchanged: NaNs sort last ascending."""
    nan = float("nan")
    arr = ak.Array([[1.0, nan, 2.0, nan, 0.5]])
    result = ak.sort(arr, ascending=True)
    values = result.tolist()[0]
    # NaNs go last; non-NaN values sorted ascending
    non_nan = [v for v in values if not (v != v)]
    nans = [v for v in values if v != v]
    assert non_nan == sorted(non_nan)
    assert len(nans) == 2


def test_sort_bool_ascending():
    """Bool sort ascending: False < True."""
    arr = ak.Array([[True, False, True, False]])
    result = ak.sort(arr, ascending=True)
    assert result.tolist() == [[False, False, True, True]]
