# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_issue434():
    a = ak.Array([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5]])
    b = ak.Array([[9.9, 8.8, 7.7], [6.6, 5.5], [4.4]])
    assert ak.to_list(b[ak.argmin(a, axis=1, keepdims=True)]) == [[9.9], [6.6], [4.4]]
    assert ak.to_list(b[ak.argmax(a, axis=1, keepdims=True)]) == [[7.7], [5.5], [4.4]]


def test_nokeepdims():
    nparray = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert ak.to_list(regular_regular) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert ak.to_list(listoffset_regular) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert ak.to_list(regular_listoffset) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert ak.to_list(listoffset_listoffset) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * var * int64"
    axis1 = ak.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.sum(listoffset_listoffset, axis=-3)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * 5 * int64"
    axis1 = ak.sum(listoffset_regular, axis=-1)
    axis2 = ak.sum(listoffset_regular, axis=-2)
    axis3 = ak.sum(listoffset_regular, axis=-3)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 5 * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * 5 * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * var * int64"
    axis1 = ak.sum(regular_listoffset, axis=-1)
    axis2 = ak.sum(regular_listoffset, axis=-2)
    axis3 = ak.sum(regular_listoffset, axis=-3)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * 5 * int64"
    axis1 = ak.sum(regular_regular, axis=-1)
    axis2 = ak.sum(regular_regular, axis=-2)
    axis3 = ak.sum(regular_regular, axis=-3)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 5 * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * 5 * int64"


def test_keepdims():
    nparray = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * var * int64"
    axis1 = ak.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * 5 * int64"
    axis1 = ak.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_regular, axis=-3, keepdims=True)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * var * 1 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * 5 * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * 5 * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * var * int64"
    axis1 = ak.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_listoffset, axis=-3, keepdims=True)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 1 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * 3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * 5 * int64"
    axis1 = ak.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_regular, axis=-3, keepdims=True)
    assert ak.to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert ak.to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert ak.to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * 1 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 1 * 5 * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * 3 * 5 * int64"


def test_nokeepdims_none1():
    content = ak.Array(
        [
            0,
            1,
            2,
            None,
            4,
            5,
            None,
            None,
            8,
            9,
            10,
            11,
            12,
            None,
            14,
            15,
            16,
            17,
            18,
            None,
            None,
            None,
            None,
            None,
            None,
            25,
            26,
            27,
            28,
            29,
        ]
    ).layout
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * var * ?int64"
    axis1 = ak.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.sum(listoffset_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * 5 * ?int64"
    axis1 = ak.sum(listoffset_regular, axis=-1)
    axis2 = ak.sum(listoffset_regular, axis=-2)
    axis3 = ak.sum(listoffset_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * var * ?int64"
    axis1 = ak.sum(regular_listoffset, axis=-1)
    axis2 = ak.sum(regular_listoffset, axis=-2)
    axis3 = ak.sum(regular_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * 5 * ?int64"
    axis1 = ak.sum(regular_regular, axis=-1)
    axis2 = ak.sum(regular_regular, axis=-2)
    axis3 = ak.sum(regular_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"


def test_keepdims_none1():
    content = ak.Array(
        [
            0,
            1,
            2,
            None,
            4,
            5,
            None,
            None,
            8,
            9,
            10,
            11,
            12,
            None,
            14,
            15,
            16,
            17,
            18,
            None,
            None,
            None,
            None,
            None,
            None,
            25,
            26,
            27,
            28,
            29,
        ]
    ).layout
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * var * ?int64"
    axis1 = ak.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * 5 * ?int64"
    axis1 = ak.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * var * ?int64"
    axis1 = ak.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 1 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * 3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * 5 * ?int64"
    axis1 = ak.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 1 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * 3 * var * int64"


def test_nokeepdims_mask1():
    mask = ak.layout.Index8(
        np.array(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ]
        )
    )
    content = ak.layout.ByteMaskedArray(
        mask,
        ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
        valid_when=False,
    )
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * var * ?int64"
    axis1 = ak.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.sum(listoffset_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * 5 * ?int64"
    axis1 = ak.sum(listoffset_regular, axis=-1)
    axis2 = ak.sum(listoffset_regular, axis=-2)
    axis3 = ak.sum(listoffset_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * var * ?int64"
    axis1 = ak.sum(regular_listoffset, axis=-1)
    axis2 = ak.sum(regular_listoffset, axis=-2)
    axis3 = ak.sum(regular_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * 5 * ?int64"
    axis1 = ak.sum(regular_regular, axis=-1)
    axis2 = ak.sum(regular_regular, axis=-2)
    axis3 = ak.sum(regular_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"


def test_keepdims_mask1():
    mask = ak.layout.Index8(
        np.array(
            [
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ]
        )
    )
    content = ak.layout.ByteMaskedArray(
        mask,
        ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
        valid_when=False,
    )
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * var * ?int64"
    axis1 = ak.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * 5 * ?int64"
    axis1 = ak.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * var * ?int64"
    axis1 = ak.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 1 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * 3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * 5 * ?int64"
    axis1 = ak.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * 3 * var * int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * 1 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * 3 * var * int64"


def test_nokeepdims_mask2():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    mask = ak.layout.Index8(np.array([False, False, True, True, False, True]))
    regular_regular = ak.layout.RegularArray(
        ak.layout.ByteMaskedArray(mask, regular, valid_when=False), 3
    )
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(
        ak.layout.ByteMaskedArray(mask, listoffset, valid_when=False), 3
    )
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert (
        str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * option[var * int64]"
    )
    axis1 = ak.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.sum(listoffset_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * ?int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * option[5 * int64]"
    axis1 = ak.sum(listoffset_regular, axis=-1)
    axis2 = ak.sum(listoffset_regular, axis=-2)
    axis3 = ak.sum(listoffset_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * ?int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * option[var * int64]"
    axis1 = ak.sum(regular_listoffset, axis=-1)
    axis2 = ak.sum(regular_listoffset, axis=-2)
    axis3 = ak.sum(regular_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * ?int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * option[5 * int64]"
    axis1 = ak.sum(regular_regular, axis=-1)
    axis2 = ak.sum(regular_regular, axis=-2)
    axis3 = ak.sum(regular_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * ?int64"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"


def test_keepdims_mask2():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    mask = ak.layout.Index8(np.array([False, False, True, True, False, True]))
    regular_regular = ak.layout.RegularArray(
        ak.layout.ByteMaskedArray(mask, regular, valid_when=False), 3
    )
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(
        ak.layout.ByteMaskedArray(mask, listoffset, valid_when=False), 3
    )
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)

    assert (
        str(ak.type(ak.Array(listoffset_listoffset))) == "2 * var * option[var * int64]"
    )
    axis1 = ak.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * option[var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * var * option[5 * int64]"
    axis1 = ak.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * option[1 * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * 3 * option[var * int64]"
    axis1 = ak.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * option[var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * 3 * option[5 * int64]"
    axis1 = ak.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * var * option[1 * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * var * var * int64"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"


def test_nokeepdims_mask3():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)
    mask = ak.layout.Index8(np.array([True, False]))
    regular_regular = ak.layout.ByteMaskedArray(mask, regular_regular, valid_when=False)
    listoffset_regular = ak.layout.ByteMaskedArray(
        mask, listoffset_regular, valid_when=False
    )
    regular_listoffset = ak.layout.ByteMaskedArray(
        mask, regular_listoffset, valid_when=False
    )
    listoffset_listoffset = ak.layout.ByteMaskedArray(
        mask, listoffset_listoffset, valid_when=False
    )

    assert (
        str(ak.type(ak.Array(listoffset_listoffset))) == "2 * option[var * var * int64]"
    )
    axis1 = ak.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.sum(listoffset_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[var * int64]"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * option[var * 5 * int64]"
    axis1 = ak.sum(listoffset_regular, axis=-1)
    axis2 = ak.sum(listoffset_regular, axis=-2)
    axis3 = ak.sum(listoffset_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[5 * int64]"
    assert str(ak.type(ak.Array(axis3))) == "3 * 5 * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * option[3 * var * int64]"
    axis1 = ak.sum(regular_listoffset, axis=-1)
    axis2 = ak.sum(regular_listoffset, axis=-2)
    axis3 = ak.sum(regular_listoffset, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[var * int64]"
    assert str(ak.type(ak.Array(axis3))) == "3 * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * option[3 * 5 * int64]"
    axis1 = ak.sum(regular_regular, axis=-1)
    axis2 = ak.sum(regular_regular, axis=-2)
    axis3 = ak.sum(regular_regular, axis=-3)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[5 * int64]"
    assert str(ak.type(ak.Array(axis3))) == "3 * 5 * int64"


def test_keepdims_mask3():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.layout.RegularArray(content, 5)
    listoffset = regular.toListOffsetArray64(False)
    regular_regular = ak.layout.RegularArray(regular, 3)
    listoffset_regular = regular_regular.toListOffsetArray64(False)
    regular_listoffset = ak.layout.RegularArray(listoffset, 3)
    listoffset_listoffset = regular_listoffset.toListOffsetArray64(False)
    mask = ak.layout.Index8(np.array([True, False]))
    regular_regular = ak.layout.ByteMaskedArray(mask, regular_regular, valid_when=False)
    listoffset_regular = ak.layout.ByteMaskedArray(
        mask, listoffset_regular, valid_when=False
    )
    regular_listoffset = ak.layout.ByteMaskedArray(
        mask, regular_listoffset, valid_when=False
    )
    listoffset_listoffset = ak.layout.ByteMaskedArray(
        mask, listoffset_listoffset, valid_when=False
    )

    assert (
        str(ak.type(ak.Array(listoffset_listoffset))) == "2 * option[var * var * int64]"
    )
    axis1 = ak.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[var * var * int64]"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(listoffset_regular))) == "2 * option[var * 5 * int64]"
    axis1 = ak.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * 1 * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[var * 5 * int64]"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * 5 * int64"

    assert str(ak.type(ak.Array(regular_listoffset))) == "2 * option[3 * var * int64]"
    axis1 = ak.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * var * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[var * var * int64]"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * var * int64"

    assert str(ak.type(ak.Array(regular_regular))) == "2 * option[3 * 5 * int64]"
    axis1 = ak.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.type(ak.Array(axis1))) == "2 * option[var * 1 * int64]"
    assert str(ak.type(ak.Array(axis2))) == "2 * option[var * 5 * int64]"
    assert str(ak.type(ak.Array(axis3))) == "1 * var * 5 * int64"
