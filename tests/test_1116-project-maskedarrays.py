# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_bytemaskedarray():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask2 = ak.index.Index8(np.array([0, 1, 0, 0], dtype=np.int8))
    maskedarray2 = ak.contents.ByteMaskedArray(mask2, array, valid_when=False)

    assert to_list(maskedarray2.project()) == [
        [0.0, 1.1, 2.2],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]

    assert to_list(maskedarray2.project(mask2)) == [
        [0.0, 1.1, 2.2],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]


def test_bitmaskedarray():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.IndexU8(np.array([0, 1, 0, 0], dtype=np.uint8))
    maskedarray2 = ak.contents.BitMaskedArray(
        mask, array, valid_when=False, length=4, lsb_order=True
    )

    assert to_list(maskedarray2.project()) == [
        [0.0, 1.1, 2.2],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]


def test_unmasked():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    unmaskedarray2 = ak.contents.UnmaskedArray(array)

    assert to_list(unmaskedarray2.project()) == [
        [0.0, 1.1, 2.2],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]


def test_indexed():
    array2 = ak.highlevel.Array([1, 2, 3, None, 4, None, None, 5]).layout
    mask2 = ak.index.Index8(np.array([0, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))

    assert to_list(array2.project()) == [1, 2, 3, 4, 5]
    assert to_list(array2.project(mask2)) == [1, 3]
