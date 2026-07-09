# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Targeted coverage for awkward_sort / awkward_argsort edge cases that the
data-driven kernel-test generator cannot exercise (these kernels are excluded
from automatic test generation):

  * NaN ordering: the CPU comparator treats NaN as "less than everything", so
    NaNs are pushed to the LOW end (front) of each sublist regardless of
    ascending/descending. This differs from NumPy, which sorts NaNs to the
    high end. argsort follows the same ordering.
  * Empty sublists: a zero-length segment must produce a zero-length output
    segment, and must not perturb neighbouring segments.
"""

from __future__ import annotations

import math

import awkward as ak

NAN = float("nan")


def _nan_eq(a, b):
    """Nested-list equality where NaN == NaN (so float results are comparable)."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _nan_eq(x, y) for x, y in zip(a, b, strict=True)
        )
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    return a == b


# ----------------------------------------------------------------------------
# sort: NaN pushed to the low end, in both directions
# ----------------------------------------------------------------------------


def test_sort_nan_ascending_low_end():
    array = ak.Array([[3.0, NAN, 1.0], [2.0, NAN]])
    result = ak.sort(array, axis=-1, ascending=True).to_list()
    assert _nan_eq(result, [[NAN, 1.0, 3.0], [NAN, 2.0]])


def test_sort_nan_descending_low_end():
    array = ak.Array([[3.0, NAN, 1.0], [2.0, NAN]])
    result = ak.sort(array, axis=-1, ascending=False).to_list()
    assert _nan_eq(result, [[NAN, 3.0, 1.0], [NAN, 2.0]])


def test_sort_all_nan_unchanged():
    array = ak.Array([[NAN, NAN], [NAN]])
    result = ak.sort(array, axis=-1).to_list()
    assert _nan_eq(result, [[NAN, NAN], [NAN]])


def test_sort_nan_with_empty_segments():
    # An empty sublist between non-empty ones must stay empty and not shift
    # the NaN-low ordering of its neighbours.
    array = ak.Array([[3.0, NAN, 1.0], [], [NAN, 2.0]])
    result = ak.sort(array, axis=-1, ascending=True).to_list()
    assert _nan_eq(result, [[NAN, 1.0, 3.0], [], [NAN, 2.0]])


# ----------------------------------------------------------------------------
# argsort: indices are local to each sublist and follow NaN-low ordering
# ----------------------------------------------------------------------------


def test_argsort_nan_ascending():
    array = ak.Array([[3.0, NAN, 1.0], [2.0, NAN]])
    # [3.0, NaN, 1.0] -> NaN(1), 1.0(2), 3.0(0); [2.0, NaN] -> NaN(1), 2.0(0)
    assert ak.argsort(array, axis=-1, ascending=True).to_list() == [[1, 2, 0], [1, 0]]


def test_argsort_nan_descending():
    array = ak.Array([[3.0, NAN, 1.0], [2.0, NAN]])
    # NaN still leads; remaining values descending.
    assert ak.argsort(array, axis=-1, ascending=False).to_list() == [[1, 0, 2], [1, 0]]


def test_argsort_nan_with_empty_segments():
    array = ak.Array([[3.0, NAN, 1.0], [], [NAN, 2.0]])
    # [NAN, 2.0] -> NaN(0) leads, then 2.0(1).
    assert ak.argsort(array, axis=-1, ascending=True).to_list() == [
        [1, 2, 0],
        [],
        [0, 1],
    ]


# ----------------------------------------------------------------------------
# empty-segment handling on non-float data
# ----------------------------------------------------------------------------


def test_sort_all_empty_segments():
    array = ak.Array([[], [], []])
    assert ak.sort(array, axis=-1).to_list() == [[], [], []]
    assert ak.argsort(array, axis=-1).to_list() == [[], [], []]


def test_sort_empty_segments_interleaved():
    array = ak.Array([[3, 1, 2], [], [5], [], [9, 4]])
    assert ak.sort(array, axis=-1).to_list() == [[1, 2, 3], [], [5], [], [4, 9]]
    assert ak.argsort(array, axis=-1).to_list() == [[1, 2, 0], [], [0], [], [1, 0]]


def test_argsort_empty_then_nonempty():
    # Leading empty segment must not offset the local indices of later ones.
    array = ak.Array([[], [2.0, 0.0, 1.0]])
    assert ak.argsort(array, axis=-1, ascending=True).to_list() == [[], [1, 2, 0]]
