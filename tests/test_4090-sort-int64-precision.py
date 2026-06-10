# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""
Regression tests for PR #4090: int64 precision loss in sort comparators.

Before the fix, sort_order_ascending/descending cast both operands to double
and called std::isnan for every comparison. For int64/uint64 values beyond
2^53, double can no longer represent consecutive integers distinctly, so
values that differ only below the double precision threshold compared as equal,
breaking strict-weak-ordering and producing wrong sort results.
"""

import numpy as np
import pytest

import awkward as ak


def test_sort_int64_above_2_53_ascending():
    """ak.sort on int64 values > 2^53 must match numpy, ascending."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.sort(arr, ascending=True)
    expected = np.sort(np.array(vals, dtype=np.int64)).tolist()
    assert result.tolist()[0] == expected


def test_sort_int64_above_2_53_descending():
    """ak.sort on int64 values > 2^53 must match numpy, descending."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.sort(arr, ascending=False)
    expected = np.sort(np.array(vals, dtype=np.int64))[::-1].tolist()
    assert result.tolist()[0] == expected


def test_sort_int64_above_2_53_stable():
    """ak.sort stable on int64 values > 2^53 must match numpy stable sort."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.sort(arr, ascending=True, stable=True)
    expected = np.sort(np.array(vals, dtype=np.int64), kind="stable").tolist()
    assert result.tolist()[0] == expected


def test_argsort_int64_above_2_53_ascending():
    """ak.argsort on int64 values > 2^53 must match numpy, ascending."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.argsort(arr, ascending=True)
    expected = np.argsort(np.array(vals, dtype=np.int64), kind="stable").tolist()
    assert result.tolist()[0] == expected


def test_argsort_int64_above_2_53_descending():
    """ak.argsort on int64 values > 2^53 must match numpy reversed, descending."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.argsort(arr, ascending=False)
    expected = np.argsort(np.array(vals, dtype=np.int64))[::-1].tolist()
    assert result.tolist()[0] == expected


def test_argsort_int64_above_2_53_stable():
    """ak.argsort stable on int64 values > 2^53 must match numpy stable argsort."""
    base = 2**60
    vals = [base + 3, base + 1, base + 2, base + 0]
    arr = ak.Array([vals])
    result = ak.argsort(arr, ascending=True, stable=True)
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


def test_sort_float_nan_descending():
    """NaN handling for float sort, descending order."""
    nan = float("nan")
    arr = ak.Array([[1.0, nan, 2.0, nan, 0.5]])
    result = ak.sort(arr, ascending=False)
    values = result.tolist()[0]
    non_nan = [v for v in values if not (v != v)]
    nans = [v for v in values if v != v]
    assert non_nan == sorted(non_nan, reverse=True)
    assert len(nans) == 2


def test_sort_bool_ascending():
    """Bool sort ascending: False < True."""
    arr = ak.Array([[True, False, True, False]])
    result = ak.sort(arr, ascending=True)
    assert result.tolist() == [[False, False, True, True]]


def test_sort_bool_descending():
    """Bool sort descending: True > False."""
    arr = ak.Array([[True, False, True, False]])
    result = ak.sort(arr, ascending=False)
    assert result.tolist() == [[True, True, False, False]]


def test_argsort_bool_ascending():
    """Bool argsort ascending."""
    arr = ak.Array([[True, False, True, False]])
    result = ak.argsort(arr, ascending=True)
    # False indices come first
    for idx in result.tolist()[0][:2]:
        assert not arr[0][idx]
    for idx in result.tolist()[0][2:]:
        assert arr[0][idx]


def test_sort_multiple_sublists():
    """Sort/argsort must handle multiple sublists correctly."""
    base = 2**60
    arr = ak.Array(
        [
            [base + 3, base + 1, base + 2],
            [base + 0, base + 4, base + 2],
        ]
    )
    result = ak.sort(arr, ascending=True)
    for i, sublist in enumerate(result.tolist()):
        assert sublist == sorted(sublist)

    result_args = ak.argsort(arr, ascending=True)
    for i, (orig, args) in enumerate(zip(arr.tolist(), result_args.tolist())):
        reordered = [orig[j] for j in args]
        assert reordered == sorted(orig)
