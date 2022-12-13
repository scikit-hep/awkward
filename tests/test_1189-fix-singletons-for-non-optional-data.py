# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    ak_array = ak.Array([1, 2, 3])
    assert ak.operations.singletons(ak_array).to_list() == [[1], [2], [3]]


# FIXME: action to be taken on deciding between [None] [] ; see Issue #983
def test2():
    a = ak.Array([[3, 1, 2], [4, 5], []])
    assert ak.operations.argmin(a, axis=1, keepdims=True).to_list() == [
        [1],
        [0],
        [None],
    ]
    assert ak.operations.singletons(ak.operations.argmin(a, axis=1)).to_list() == [
        [1],
        [0],
        [],
    ]

    assert a[ak.operations.argmin(a, axis=1, keepdims=True)].to_list() == [
        [1],
        [4],
        [None],
    ]
    assert a[ak.operations.singletons(ak.operations.argmin(a, axis=1))].to_list() == [
        [1],
        [4],
        [],
    ]


def test_numpyarray():
    a = ak.contents.NumpyArray(np.arange(12).reshape(4, 3))
    assert ak.operations.singletons(a, axis=1).to_list() == [
        [[0], [1], [2]],
        [[3], [4], [5]],
        [[6], [7], [8]],
        [[9], [10], [11]],
    ]


def test_empyarray():
    e = ak.contents.EmptyArray()
    assert ak.operations.singletons(e).to_list() == []
