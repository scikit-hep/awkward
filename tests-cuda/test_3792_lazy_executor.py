# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


@pytest.fixture
def arr():
    return ak.Array([[1, 2, 3], [4, 5], [6, 7, 8, 9]], backend="cuda")


def test_arithmetic_chain(arr):
    lazy_arr = ak.cuda.lazy(arr)
    transformed = lazy_arr * 2 + 1
    assert ak.to_list(transformed.compute()) == [
        [3, 5, 7],
        [9, 11],
        [13, 15, 17, 19],
    ]


def test_filter_on_input_condition(arr):
    lazy_arr = ak.cuda.lazy(arr)
    transformed = lazy_arr * 2 + 1
    result = transformed.filter(lazy_arr > 3)
    assert ak.to_list(result.compute()) == [[], [9, 11], [13, 15, 17, 19]]


def test_filter_on_transformed_condition(arr):
    lazy_arr = ak.cuda.lazy(arr)
    transformed = lazy_arr * 2 + 1
    result = transformed.filter(transformed > 5)
    assert ak.to_list(result.compute()) == [[7], [9, 11], [13, 15, 17, 19]]


def test_visualize_is_string(arr):
    lazy_arr = ak.cuda.lazy(arr)
    result = (lazy_arr * 2 + 1).filter(lazy_arr > 3)
    text = result.visualize()
    assert isinstance(text, str)
    assert "filter" in text


def test_combinations():
    arr2 = ak.Array([[1, 2, 3], [4, 5]], backend="cuda")
    pairs = ak.cuda.lazy(arr2).combinations(2)
    assert ak.to_list(pairs.compute()) == [[(1, 2), (1, 3), (2, 3)], [(4, 5)]]


def test_field_access_on_combinations():
    arr3 = ak.Array([[1, 2, 3], [4, 5]], backend="cuda")
    pairs = ak.cuda.lazy(arr3).combinations(2)
    pair_sums = pairs["0"] + pairs["1"]
    assert ak.to_list(pair_sums.compute()) == [[3, 4, 5], [9]]


def test_select_lists(arr):
    lazy_arr = ak.cuda.lazy(arr)
    result = lazy_arr.select_lists(cp.array([True, False, True]))
    assert ak.to_list(result.compute()) == [[1, 2, 3], [6, 7, 8, 9]]


def test_compute_is_memoized(arr):
    lazy_arr = ak.cuda.lazy(arr)
    result = (lazy_arr * 2 + 1).filter(lazy_arr > 3)
    first = result.compute()
    second = result.compute()
    assert first is second


def test_cpu_backed_array_is_rejected():
    cpu_arr = ak.Array([[1, 2], [3]])
    with pytest.raises(TypeError):
        ak.cuda.lazy(cpu_arr)
