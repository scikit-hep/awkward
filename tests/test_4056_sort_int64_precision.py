# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak

# Values beyond 2^53 are not exactly representable as float64: a comparator
# that casts int64/uint64 through double would compare distinct integers as
# equal (breaking strict weak ordering) and produce wrong sort orders.
BIG = 2**53

INT64_VALUES = [BIG + 1, BIG, BIG + 3, BIG + 2, BIG - 1]


def test_sort_int64_beyond_2_53_ascending():
    array = ak.Array([INT64_VALUES])
    assert ak.sort(array, axis=-1).to_list() == [sorted(INT64_VALUES)]


def test_sort_int64_beyond_2_53_descending():
    array = ak.Array([INT64_VALUES])
    assert ak.sort(array, axis=-1, ascending=False).to_list() == [
        sorted(INT64_VALUES, reverse=True)
    ]


def test_sort_int64_beyond_2_53_stable():
    array = ak.Array([INT64_VALUES])
    assert ak.sort(array, axis=-1, stable=True).to_list() == [sorted(INT64_VALUES)]


def test_argsort_int64_beyond_2_53_ascending():
    array = ak.Array([INT64_VALUES])
    expected = np.argsort(np.array(INT64_VALUES, dtype=np.int64), kind="stable")
    assert ak.argsort(array, axis=-1).to_list() == [expected.tolist()]


def test_argsort_int64_beyond_2_53_descending():
    array = ak.Array([INT64_VALUES])
    expected = np.argsort(-np.array(INT64_VALUES, dtype=np.int64), kind="stable")
    assert ak.argsort(array, axis=-1, ascending=False).to_list() == [expected.tolist()]


def test_argsort_int64_adjacent_values_stable():
    # BIG and BIG + 1 round to the same float64; a double-casting comparator
    # would consider them equal, and a *stable* sort would then keep the
    # original (wrong) order. The exact integer comparator must swap them.
    array = ak.Array([[BIG + 1, BIG]])
    assert ak.argsort(array, axis=-1, stable=True).to_list() == [[1, 0]]
    assert ak.sort(array, axis=-1, stable=True).to_list() == [[BIG, BIG + 1]]


def test_sort_uint64_beyond_2_53():
    values = np.array([2**64 - 1, 2**64 - 2, BIG + 1, BIG], dtype=np.uint64)
    array = ak.unflatten(ak.Array(values), [4])
    assert ak.sort(array, axis=-1).to_list() == [np.sort(values).tolist()]
    assert ak.sort(array, axis=-1, ascending=False).to_list() == [
        np.sort(values)[::-1].tolist()
    ]


def test_argsort_uint64_beyond_2_53():
    values = np.array([2**64 - 1, 2**64 - 2, BIG + 1, BIG], dtype=np.uint64)
    array = ak.unflatten(ak.Array(values), [4])
    expected = np.argsort(values, kind="stable")
    assert ak.argsort(array, axis=-1).to_list() == [expected.tolist()]


def test_sort_float_nan_ascending_unchanged():
    # NaN handling is unchanged: NaNs compare "less" than everything and are
    # pushed to the low end (note: NumPy puts them at the high end instead).
    array = ak.Array([[3.0, float("nan"), 1.0, 2.0]])
    result = ak.sort(array, axis=-1).to_list()[0]
    assert np.isnan(result[0])
    assert result[1:] == [1.0, 2.0, 3.0]


def test_sort_float_nan_descending_unchanged():
    array = ak.Array([[3.0, float("nan"), 1.0, 2.0]])
    result = ak.sort(array, axis=-1, ascending=False).to_list()[0]
    assert np.isnan(result[0])
    assert result[1:] == [3.0, 2.0, 1.0]


def test_argsort_float_nan_unchanged():
    array = ak.Array([[3.0, float("nan"), 1.0, 2.0]])
    assert ak.argsort(array, axis=-1).to_list() == [[1, 2, 3, 0]]


def test_sort_bool():
    array = ak.Array([[True, False, True, False]])
    assert ak.sort(array, axis=-1).to_list() == [[False, False, True, True]]
    assert ak.sort(array, axis=-1, ascending=False).to_list() == [
        [True, True, False, False]
    ]


def test_sort_int64_beyond_2_53_multiple_sublists():
    data = [
        [BIG + 2, BIG, BIG + 1],
        [],
        [BIG + 5, BIG + 4],
        [BIG - 1],
        [BIG + 7, BIG + 6, BIG + 9, BIG + 8],
    ]
    array = ak.Array(data)
    assert ak.sort(array, axis=-1).to_list() == [sorted(row) for row in data]
    expected_args = [
        np.argsort(np.array(row, dtype=np.int64), kind="stable").tolist()
        for row in data
    ]
    assert ak.argsort(array, axis=-1).to_list() == expected_args
