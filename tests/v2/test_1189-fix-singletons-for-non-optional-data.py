# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    ak_array = ak._v2.Array([1, 2, 3])
    assert ak._v2.operations.singletons(ak_array).tolist() == [[1], [2], [3]]


# FIXME: action to be taken on deciding between [None] [] ; see Issue #983
def test2():
    a = ak._v2.Array([[3, 1, 2], [4, 5], []])
    assert ak._v2.operations.argmin(a, axis=1, keepdims=True).tolist() == [
        [1],
        [0],
        [None],
    ]
    assert ak._v2.operations.singletons(
        ak._v2.operations.argmin(a, axis=1)
    ).tolist() == [[1], [0], []]

    assert a[ak._v2.operations.argmin(a, axis=1, keepdims=True)].tolist() == [
        [1],
        [4],
        [None],
    ]
    assert a[
        ak._v2.operations.singletons(ak._v2.operations.argmin(a, axis=1))
    ].tolist() == [[1], [4], []]


def test_numpyarray():
    a = ak._v2.contents.NumpyArray(np.arange(12).reshape(4, 3))
    assert ak._v2.operations.singletons(a).tolist() == [
        [[0], [1], [2]],
        [[3], [4], [5]],
        [[6], [7], [8]],
        [[9], [10], [11]],
    ]


def test_empyarray():
    e = ak._v2.contents.EmptyArray()
    assert ak._v2.operations.singletons(e).tolist() == []
