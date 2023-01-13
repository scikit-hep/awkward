# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak.numba.register_and_check()


def test_ByteMaskedArray():
    content = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.highlevel.Array(
        ak.contents.ByteMaskedArray(mask, content, valid_when=False)
    )
    assert ak.operations.to_list(array) == [
        [0.0, 1.1, 2.2],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, ak.contents.ByteMaskedArray)
    assert ak.operations.to_list(y) == ak.operations.to_list(array)

    @numba.njit
    def f3(x, i):
        return x[i]

    assert ak.operations.to_list(f3(array, 0)) == [0.0, 1.1, 2.2]
    assert ak.operations.to_list(f3(array, 1)) == []
    assert f3(array, 2) is None
    assert f3(array, 3) is None
    assert ak.operations.to_list(f3(array, 4)) == [6.6, 7.7, 8.8, 9.9]


def test_BitMaskedArray():
    content = ak.contents.NumpyArray(np.arange(13))
    mask = ak.index.IndexU8(np.array([58, 59], dtype=np.uint8))
    array = ak.highlevel.Array(
        ak.contents.BitMaskedArray(
            mask, content, valid_when=True, length=13, lsb_order=True
        )
    )
    assert ak.operations.to_list(array) == [
        None,
        1,
        None,
        3,
        4,
        5,
        None,
        None,
        8,
        9,
        None,
        11,
        12,
    ]

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, ak.contents.ByteMaskedArray)
    assert ak.operations.to_list(y) == ak.operations.to_list(array)

    @numba.njit
    def f3(x, i):
        return x[i]

    assert [f3(array, i) for i in range(len(array))] == [
        None,
        1,
        None,
        3,
        4,
        5,
        None,
        None,
        8,
        9,
        None,
        11,
        12,
    ]

    array = ak.highlevel.Array(
        ak.contents.BitMaskedArray(
            mask, content, valid_when=True, length=13, lsb_order=False
        )
    )
    assert ak.operations.to_list(array) == [
        None,
        None,
        2,
        3,
        4,
        None,
        6,
        None,
        None,
        None,
        10,
        11,
        12,
    ]

    assert [f3(array, i) for i in range(len(array))] == [
        None,
        None,
        2,
        3,
        4,
        None,
        6,
        None,
        None,
        None,
        10,
        11,
        12,
    ]


def test_UnmaskedArray():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    )
    array = ak.highlevel.Array(ak.contents.UnmaskedArray(content))
    assert ak.operations.to_list(array) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert str(array.type) == "5 * ?float64"

    @numba.njit
    def f1(x):
        return 3.14

    f1(array)

    @numba.njit
    def f2(x):
        return x

    y = f2(array)
    assert isinstance(y.layout, ak.contents.UnmaskedArray)
    assert ak.operations.to_list(y) == ak.operations.to_list(array)
    assert str(y.type) == str(array.type)

    @numba.njit
    def f3(x, i):
        return x[i]

    assert [f3(array, i) for i in range(len(array))] == [1.1, 2.2, 3.3, 4.4, 5.5]
