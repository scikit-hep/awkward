# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2, v1v2_equal, v1_to_v2_index


def test():
    data = np.array([0, 2, 2, 3], dtype="i8")
    offsets = ak.layout.Index64(data)
    offsets2 = v1_to_v2_index(offsets)

    assert np.asarray(offsets2).tolist() == [0, 2, 2, 3]
    assert offsets2[0] == 0
    assert offsets2[1] == 2
    assert offsets2[2] == 2
    assert offsets2[3] == 3
    data[2] = 999
    assert offsets2[2] == 999

    assert np.asarray(offsets2).tolist() == np.asarray(offsets).tolist()
    assert offsets2[0] == offsets2[0]
    assert offsets2[1] == offsets2[1]
    assert offsets2[2] == offsets2[2]
    assert offsets2[3] == offsets2[3]

    data = np.array([0, 2, 2, 3], dtype="i4")
    offsets = ak.layout.Index32(data)
    offsets2 = v1_to_v2_index(offsets)

    assert np.asarray(offsets2).tolist() == [0, 2, 2, 3]
    assert offsets2[0] == 0
    assert offsets2[1] == 2
    assert offsets2[2] == 2
    assert offsets2[3] == 3
    data[2] = 999
    assert offsets2[2] == 999

    assert np.asarray(offsets2).tolist() == np.asarray(offsets).tolist()
    assert offsets2[0] == offsets2[0]
    assert offsets2[1] == offsets2[1]
    assert offsets2[2] == offsets2[2]
    assert offsets2[3] == offsets2[3]

    content = ak.layout.NumpyArray(np.arange(12).reshape(3, 4))
    content2 = v1_to_v2(content)

    assert np.asarray(content2).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert np.asarray(content2[0]).tolist() == [0, 1, 2, 3]
    assert np.asarray(content2[1]).tolist() == [4, 5, 6, 7]
    assert np.asarray(content2[2]).tolist() == [8, 9, 10, 11]
    assert [content2[i][j] for i in range(3) for j in range(4)] == [
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

    assert np.asarray(content2).tolist() == np.asarray(content).tolist()
    assert np.asarray(content2[0]).tolist() == np.asarray(content[0]).tolist()
    assert np.asarray(content2[1]).tolist() == np.asarray(content[1]).tolist()
    assert np.asarray(content2[2]).tolist() == np.asarray(content[2]).tolist()
    assert [content2[i][j] for i in range(3) for j in range(4)] == [
        content[i][j] for i in range(3) for j in range(4)
    ]

    data = np.array([0, 2, 2, 3], dtype="i4")
    offsets = ak.layout.Index32(data)
    array = ak.layout.ListOffsetArray32(offsets, content)
    array2 = v1_to_v2(array)

    assert np.asarray(array2[0]).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert np.asarray(array2[1]).tolist() == []
    assert np.asarray(array2[2]).tolist() == [[8, 9, 10, 11]]
    assert np.asarray(array2[1:3][0]).tolist() == []
    assert np.asarray(array2[1:3][1]).tolist() == [[8, 9, 10, 11]]
    assert np.asarray(array2[2:3][0]).tolist() == [[8, 9, 10, 11]]

    assert np.asarray(array2[0]).tolist() == np.asarray(array[0]).tolist()
    assert np.asarray(array2[1]).tolist() == np.asarray(array[1]).tolist()
    assert np.asarray(array2[2]).tolist() == np.asarray(array[2]).tolist()
    assert np.asarray(array2[1:3][0]).tolist() == np.asarray(array[1:3][0]).tolist()
    assert np.asarray(array2[1:3][1]).tolist() == np.asarray(array[1:3][1]).tolist()
    assert np.asarray(array2[2:3][0]).tolist() == np.asarray(array[2:3][0]).tolist()


def test_len():
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], dtype="i4"))
    content = ak.layout.NumpyArray(np.arange(12).reshape(4, 3))
    array = ak.layout.ListOffsetArray32(offsets, content)
    content2 = v1_to_v2(content)
    array2 = v1_to_v2(array)

    assert len(content2) == 4
    assert len(array2) == 3

    assert len(content2) == len(content)
    assert len(array2) == len(array)


def test_members():
    offsets = ak.layout.Index32(np.array([0, 2, 2, 3], dtype="i4"))
    content = ak.layout.NumpyArray(np.arange(12).reshape(3, 4))
    array = ak.layout.ListOffsetArray32(offsets, content)
    array2 = v1_to_v2(array)

    assert np.asarray(array2.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(array2.content).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]

    assert np.asarray(array2.offsets).tolist() == np.asarray(array.offsets).tolist()
    assert np.asarray(array2.content).tolist() == np.asarray(array.content).tolist()

    new = ak.layout.ListOffsetArray32(offsets, array)
    new2 = v1_to_v2(new)

    assert np.asarray(new2.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(new2.content.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(new2.content.content).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]

    assert np.asarray(new2.offsets).tolist() == np.asarray(new.offsets).tolist()
    assert (
        np.asarray(new2.content.offsets).tolist()
        == np.asarray(new.content.offsets).tolist()
    )
    assert (
        np.asarray(new2.content.content).tolist()
        == np.asarray(new.content.content).tolist()
    )
