# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest
import numpy as np
import awkward1 as ak


def test():
    data = np.array([0, 2, 2, 3], dtype="i8")
    offsets = ak.layout.Index64(data)
    assert np.asarray(offsets).tolist() == [0, 2, 2, 3]
    assert offsets[0] == 0
    assert offsets[1] == 2
    assert offsets[2] == 2
    assert offsets[3] == 3
    data[2] = 999
    assert offsets[2] == 999

    data = np.array([0, 2, 2, 3], dtype="i4")
    offsets = ak.layout.Index32(data)
    assert np.asarray(offsets).tolist() == [0, 2, 2, 3]
    assert offsets[0] == 0
    assert offsets[1] == 2
    assert offsets[2] == 2
    assert offsets[3] == 3
    data[2] = 999
    assert offsets[2] == 999

    content = ak.layout.NumpyArray(np.arange(12).reshape(3, 4))
    assert np.asarray(content).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert np.asarray(content[0]).tolist() == [0, 1, 2, 3]
    assert np.asarray(content[1]).tolist() == [4, 5, 6, 7]
    assert np.asarray(content[2]).tolist() == [8, 9, 10, 11]
    assert [content[i][j] for i in range(3) for j in range(4)] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]

    data = np.array([0, 2, 2, 3], dtype="i4")
    offsets = ak.layout.Index32(data)
    array = ak.layout.ListOffsetArray32(offsets, content)
    assert np.asarray(array[0]).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert np.asarray(array[1]).tolist() == []
    assert np.asarray(array[2]).tolist() == [[8, 9, 10, 11]]
    assert np.asarray(array[1:3][0]).tolist() == []
    assert np.asarray(array[1:3][1]).tolist() == [[8, 9, 10, 11]]
    assert np.asarray(array[2:3][0]).tolist() == [[8, 9, 10, 11]]


def test_len():
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], dtype="i4"))
    content = ak.layout.NumpyArray(np.arange(12).reshape(4, 3))
    array = ak.layout.ListOffsetArray32(offsets, content)
    assert len(content) == 4
    assert len(array) == 3


def test_members():
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], dtype="i4"))
    content = ak.layout.NumpyArray(np.arange(12).reshape(3, 4))
    array = ak.layout.ListOffsetArray32(offsets, content)
    assert np.asarray(array.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(array.content).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    array2 = ak.layout.ListOffsetArray32(offsets, array)
    assert np.asarray(array2.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(array2.content.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(array2.content.content).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
