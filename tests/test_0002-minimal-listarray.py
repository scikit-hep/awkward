# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test():
    data = np.array([0, 2, 2, 3], dtype="i8")
    offsets = ak._v2.index.Index64(data)

    assert np.asarray(offsets).tolist() == [0, 2, 2, 3]
    assert offsets[0] == 0
    assert offsets[1] == 2
    assert offsets[2] == 2
    assert offsets[3] == 3
    data[2] = 999
    assert offsets[2] == 999

    data = np.array([0, 2, 2, 3], dtype="i4")
    offsets = ak._v2.index.Index32(data)

    assert np.asarray(offsets).tolist() == [0, 2, 2, 3]
    assert offsets[0] == 0
    assert offsets[1] == 2
    assert offsets[2] == 2
    assert offsets[3] == 3
    data[2] = 999
    assert offsets[2] == 999

    content = ak._v2.contents.NumpyArray(np.arange(12).reshape(3, 4))
    data = np.array([0, 2, 2, 3], dtype="i4")
    offsets = ak._v2.index.Index32(data)
    array = ak._v2.contents.ListOffsetArray(offsets, content)

    assert np.asarray(content).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert np.asarray(content[0]).tolist() == [0, 1, 2, 3]
    assert content.typetracer[0].form == content[0].form
    assert np.asarray(content[1]).tolist() == [4, 5, 6, 7]
    assert content.typetracer[1].form == content[1].form
    assert np.asarray(content[2]).tolist() == [8, 9, 10, 11]
    assert content.typetracer[2].form == content[2].form
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

    assert np.asarray(array[0]).tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert array.typetracer[0].form == array[0].form
    assert np.asarray(array[1]).tolist() == []
    assert array.typetracer[1].form == array[1].form
    assert np.asarray(array[2]).tolist() == [[8, 9, 10, 11]]
    assert array.typetracer[2].form == array[2].form
    assert np.asarray(array[1:3][0]).tolist() == []
    assert array.typetracer[1:3][0].form == array[1:3][0].form
    assert np.asarray(array[1:3][1]).tolist() == [[8, 9, 10, 11]]
    assert array.typetracer[1:3][1].form == array[1:3][1].form
    assert np.asarray(array[2:3][0]).tolist() == [[8, 9, 10, 11]]
    assert array.typetracer[2:3][0].form == array[2:3][0].form


def test_len():
    offsets = ak._v2.index.Index32(np.array([0, 2, 2, 3], dtype="i4"))
    content = ak._v2.contents.NumpyArray(np.arange(12).reshape(4, 3))
    array = ak._v2.contents.ListOffsetArray(offsets, content)

    assert len(content) == 4
    assert len(array) == 3


def test_members():
    offsets = ak._v2.index.Index32(np.array([0, 2, 2, 3], dtype="i4"))
    content = ak._v2.contents.NumpyArray(np.arange(12).reshape(3, 4))
    array = ak._v2.contents.ListOffsetArray(offsets, content)
    new = ak._v2.contents.ListOffsetArray(offsets, array)

    assert np.asarray(array.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(array.content).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]

    assert np.asarray(new.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(new.content.offsets).tolist() == [0, 2, 2, 3]
    assert np.asarray(new.content.content).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
