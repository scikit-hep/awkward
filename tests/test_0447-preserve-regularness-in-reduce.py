# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_issue434():
    a = ak.highlevel.Array([[0.0, 1.1, 2.2], [3.3, 4.4], [5.5]])
    b = ak.highlevel.Array([[9.9, 8.8, 7.7], [6.6, 5.5], [4.4]])
    assert to_list(b[ak.argmin(a, axis=1, keepdims=True)]) == [[9.9], [6.6], [4.4]]
    assert to_list(b[ak.argmax(a, axis=1, keepdims=True)]) == [[7.7], [5.5], [4.4]]


def test_nokeepdims():
    nparray = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert to_list(regular_regular) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert to_list(listoffset_regular) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert to_list(regular_listoffset) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert to_list(listoffset_listoffset) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type) == "2 * var * var * int64"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3)
    assert to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(listoffset_regular).type) == "2 * var * 5 * int64"
    axis1 = ak.operations.sum(listoffset_regular, axis=-1)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3)
    assert to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"

    assert str(ak.highlevel.Array(regular_listoffset).type) == "2 * 3 * var * int64"
    axis1 = ak.operations.sum(regular_listoffset, axis=-1)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3)
    assert to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * 5 * int64"
    axis1 = ak.operations.sum(regular_regular, axis=-1)
    axis2 = ak.operations.sum(regular_regular, axis=-2)
    axis3 = ak.operations.sum(regular_regular, axis=-3)
    assert to_list(axis1) == np.sum(nparray, axis=-1).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"


def test_keepdims():
    nparray = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type) == "2 * var * var * int64"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * var * int64"

    assert str(ak.highlevel.Array(listoffset_regular).type) == "2 * var * 5 * int64"
    axis1 = ak.operations.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3, keepdims=True)
    assert to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * 5 * int64"

    assert str(ak.highlevel.Array(regular_listoffset).type) == "2 * 3 * var * int64"
    axis1 = ak.operations.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3, keepdims=True)
    assert to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * 5 * int64"
    axis1 = ak.operations.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_regular, axis=-3, keepdims=True)
    assert to_list(axis1) == np.sum(nparray, axis=-1, keepdims=True).tolist()
    assert to_list(axis2) == np.sum(nparray, axis=-2, keepdims=True).tolist()
    assert to_list(axis3) == np.sum(nparray, axis=-3, keepdims=True).tolist()
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * 5 * int64"


def test_nokeepdims_none1():
    content = ak.highlevel.Array(
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
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type) == "2 * var * var * ?int64"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(listoffset_regular).type) == "2 * var * 5 * ?int64"
    axis1 = ak.operations.sum(listoffset_regular, axis=-1)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"

    assert str(ak.highlevel.Array(regular_listoffset).type) == "2 * 3 * var * ?int64"
    axis1 = ak.operations.sum(regular_listoffset, axis=-1)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * 5 * ?int64"
    axis1 = ak.operations.sum(regular_regular, axis=-1)
    axis2 = ak.operations.sum(regular_regular, axis=-2)
    axis3 = ak.operations.sum(regular_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"


def test_keepdims_none1():
    content = ak.highlevel.Array(
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
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type) == "2 * var * var * ?int64"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * var * int64"

    assert str(ak.highlevel.Array(listoffset_regular).type) == "2 * var * 5 * ?int64"
    axis1 = ak.operations.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * 5 * int64"

    assert str(ak.highlevel.Array(regular_listoffset).type) == "2 * 3 * var * ?int64"
    axis1 = ak.operations.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * 5 * ?int64"
    axis1 = ak.operations.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * 5 * int64"


def test_nokeepdims_mask1():
    mask = ak.index.Index8(
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
    content = ak.contents.ByteMaskedArray(
        mask,
        ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
        valid_when=False,
    )
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type) == "2 * var * var * ?int64"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(listoffset_regular).type) == "2 * var * 5 * ?int64"
    axis1 = ak.operations.sum(listoffset_regular, axis=-1)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"

    assert str(ak.highlevel.Array(regular_listoffset).type) == "2 * 3 * var * ?int64"
    axis1 = ak.operations.sum(regular_listoffset, axis=-1)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * 5 * ?int64"
    axis1 = ak.operations.sum(regular_regular, axis=-1)
    axis2 = ak.operations.sum(regular_regular, axis=-2)
    axis3 = ak.operations.sum(regular_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"


def test_keepdims_mask1():
    mask = ak.index.Index8(
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
    content = ak.contents.ByteMaskedArray(
        mask,
        ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64)),
        valid_when=False,
    )
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type) == "2 * var * var * ?int64"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * var * int64"

    assert str(ak.highlevel.Array(listoffset_regular).type) == "2 * var * 5 * ?int64"
    axis1 = ak.operations.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * 5 * int64"

    assert str(ak.highlevel.Array(regular_listoffset).type) == "2 * 3 * var * ?int64"
    axis1 = ak.operations.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * 5 * ?int64"
    axis1 = ak.operations.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * 1 * int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * 5 * int64"


def test_nokeepdims_mask2():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    mask = ak.index.Index8(np.array([False, False, True, True, False, True]))
    regular_regular = ak.contents.RegularArray(
        ak.contents.ByteMaskedArray(mask, regular, valid_when=False),
        3,
        zeros_length=0,
    )
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(
        ak.contents.ByteMaskedArray(mask, listoffset, valid_when=False),
        3,
        zeros_length=0,
    )
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type)
        == "2 * var * option[var * int64]"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * ?int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert (
        str(ak.highlevel.Array(listoffset_regular).type)
        == "2 * var * option[5 * int64]"
    )
    axis1 = ak.operations.sum(listoffset_regular, axis=-1)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * ?int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"

    assert (
        str(ak.highlevel.Array(regular_listoffset).type)
        == "2 * 3 * option[var * int64]"
    )
    axis1 = ak.operations.sum(regular_listoffset, axis=-1)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * ?int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * option[5 * int64]"
    axis1 = ak.operations.sum(regular_regular, axis=-1)
    axis2 = ak.operations.sum(regular_regular, axis=-2)
    axis3 = ak.operations.sum(regular_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * ?int64"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"


def test_keepdims_mask2():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    mask = ak.index.Index8(np.array([False, False, True, True, False, True]))
    regular_regular = ak.contents.RegularArray(
        ak.contents.ByteMaskedArray(mask, regular, valid_when=False),
        3,
        zeros_length=0,
    )
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(
        ak.contents.ByteMaskedArray(mask, listoffset, valid_when=False),
        3,
        zeros_length=0,
    )
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type)
        == "2 * var * option[var * int64]"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * option[1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * var * int64"

    assert (
        str(ak.highlevel.Array(listoffset_regular).type)
        == "2 * var * option[5 * int64]"
    )
    axis1 = ak.operations.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * var * option[1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * 5 * int64"

    assert (
        str(ak.highlevel.Array(regular_listoffset).type)
        == "2 * 3 * option[var * int64]"
    )
    axis1 = ak.operations.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * option[1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * var * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * 3 * option[5 * int64]"
    axis1 = ak.operations.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * 3 * option[1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * 1 * 5 * int64"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * 5 * int64"


def test_nokeepdims_mask3():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)
    mask = ak.index.Index8(np.array([True, False]))
    regular_regular = ak.contents.ByteMaskedArray(
        mask, regular_regular, valid_when=False
    )
    listoffset_regular = ak.contents.ByteMaskedArray(
        mask, listoffset_regular, valid_when=False
    )
    regular_listoffset = ak.contents.ByteMaskedArray(
        mask, regular_listoffset, valid_when=False
    )
    listoffset_listoffset = ak.contents.ByteMaskedArray(
        mask, listoffset_listoffset, valid_when=False
    )

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type)
        == "2 * option[var * var * int64]"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[var * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[var * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert (
        str(ak.highlevel.Array(listoffset_regular).type)
        == "2 * option[var * 5 * int64]"
    )
    axis1 = ak.operations.sum(listoffset_regular, axis=-1)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[var * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[5 * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"

    assert (
        str(ak.highlevel.Array(regular_listoffset).type)
        == "2 * option[3 * var * int64]"
    )
    axis1 = ak.operations.sum(regular_listoffset, axis=-1)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[3 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[var * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * option[3 * 5 * int64]"
    axis1 = ak.operations.sum(regular_regular, axis=-1)
    axis2 = ak.operations.sum(regular_regular, axis=-2)
    axis3 = ak.operations.sum(regular_regular, axis=-3)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[3 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[5 * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "3 * 5 * int64"


def test_keepdims_mask3():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64))
    regular = ak.contents.RegularArray(content, 5, zeros_length=0)
    listoffset = regular.to_ListOffsetArray64(False)
    regular_regular = ak.contents.RegularArray(regular, 3, zeros_length=0)
    listoffset_regular = regular_regular.to_ListOffsetArray64(False)
    regular_listoffset = ak.contents.RegularArray(listoffset, 3, zeros_length=0)
    listoffset_listoffset = regular_listoffset.to_ListOffsetArray64(False)
    mask = ak.index.Index8(np.array([True, False]))
    regular_regular = ak.contents.ByteMaskedArray(
        mask, regular_regular, valid_when=False
    )
    listoffset_regular = ak.contents.ByteMaskedArray(
        mask, listoffset_regular, valid_when=False
    )
    regular_listoffset = ak.contents.ByteMaskedArray(
        mask, regular_listoffset, valid_when=False
    )
    listoffset_listoffset = ak.contents.ByteMaskedArray(
        mask, listoffset_listoffset, valid_when=False
    )

    assert (
        str(ak.highlevel.Array(listoffset_listoffset).type)
        == "2 * option[var * var * int64]"
    )
    axis1 = ak.operations.sum(listoffset_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[var * 1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[1 * var * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * var * int64"

    assert (
        str(ak.highlevel.Array(listoffset_regular).type)
        == "2 * option[var * 5 * int64]"
    )
    axis1 = ak.operations.sum(listoffset_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(listoffset_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(listoffset_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[var * 1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[1 * 5 * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "1 * var * 5 * int64"

    assert (
        str(ak.highlevel.Array(regular_listoffset).type)
        == "2 * option[3 * var * int64]"
    )
    axis1 = ak.operations.sum(regular_listoffset, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_listoffset, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_listoffset, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[3 * 1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[1 * var * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * var * int64"

    assert str(ak.highlevel.Array(regular_regular).type) == "2 * option[3 * 5 * int64]"
    axis1 = ak.operations.sum(regular_regular, axis=-1, keepdims=True)
    axis2 = ak.operations.sum(regular_regular, axis=-2, keepdims=True)
    axis3 = ak.operations.sum(regular_regular, axis=-3, keepdims=True)
    assert str(ak.highlevel.Array(axis1).type) == "2 * option[3 * 1 * int64]"
    assert str(ak.highlevel.Array(axis2).type) == "2 * option[1 * 5 * int64]"
    assert str(ak.highlevel.Array(axis3).type) == "1 * 3 * 5 * int64"
