# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

to_list = ak._v2.operations.to_list


def test_rpad_and_clip_empty_array():
    empty = ak._v2.contents.emptyarray.EmptyArray()
    assert to_list(empty) == []
    assert to_list(empty.pad_none(5, 0)) == [None, None, None, None, None]
    assert empty.typetracer.pad_none(5, 0).form == empty.pad_none(5, 0).form
    assert to_list(empty.pad_none(5, 0, clip=True)) == [None, None, None, None, None]
    assert (
        empty.typetracer.pad_none(5, 0, clip=True).form
        == empty.pad_none(5, 0, clip=True).form
    )


def test_rpad_and_clip_numpy_array():
    array = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    assert to_list(array) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert to_list(array.pad_none(5, 0, clip=True)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
        None,
        None,
    ]
    assert (
        array.typetracer.pad_none(5, 0, clip=True).form
        == array.pad_none(5, 0, clip=True).form
    )

    assert to_list(array.pad_none(5, 1, clip=True)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None, None],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None, None],
    ]
    assert (
        array.typetracer.pad_none(5, 1, clip=True).form
        == array.pad_none(5, 1, clip=True).form
    )

    assert to_list(array.pad_none(7, 2, clip=True)) == [
        [
            [0, 1, 2, 3, 4, None, None],
            [5, 6, 7, 8, 9, None, None],
            [10, 11, 12, 13, 14, None, None],
        ],
        [
            [15, 16, 17, 18, 19, None, None],
            [20, 21, 22, 23, 24, None, None],
            [25, 26, 27, 28, 29, None, None],
        ],
    ]
    assert (
        array.typetracer.pad_none(7, 2, clip=True).form
        == array.pad_none(7, 2, clip=True).form
    )

    assert to_list(array.pad_none(2, 2, clip=True)) == [
        [[0, 1], [5, 6], [10, 11]],
        [[15, 16], [20, 21], [25, 26]],
    ]
    assert (
        array.typetracer.pad_none(2, 2, clip=True).form
        == array.pad_none(2, 2, clip=True).form
    )


def test_rpad_numpy_array():
    array = ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert to_list(array.pad_none(10, 0)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        None,
        None,
        None,
        None,
        None,
    ]
    assert array.typetracer.pad_none(10, 0).form == array.pad_none(10, 0).form

    array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    )
    assert to_list(array.pad_none(5, 0)) == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        None,
        None,
        None,
    ]
    assert array.typetracer.pad_none(5, 0).form == array.pad_none(5, 0).form
    assert to_list(array.pad_none(5, 1)) == [
        [1.1, 2.2, 3.3, None, None],
        [4.4, 5.5, 6.6, None, None],
    ]
    assert array.typetracer.pad_none(5, 1).form == array.pad_none(5, 1).form

    array = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    assert to_list(array) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert to_list(array.pad_none(1, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert array.typetracer.pad_none(1, 0).form == array.pad_none(1, 0).form
    assert to_list(array.pad_none(2, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.pad_none(2, 0).form == array.pad_none(2, 0).form
    assert to_list(array.pad_none(3, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
    ]
    assert array.typetracer.pad_none(3, 0).form == array.pad_none(3, 0).form
    assert to_list(array.pad_none(4, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
        None,
    ]
    assert array.typetracer.pad_none(4, 0).form == array.pad_none(4, 0).form
    assert to_list(array.pad_none(5, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
        None,
        None,
    ]
    assert array.typetracer.pad_none(5, 0).form == array.pad_none(5, 0).form

    assert to_list(array.pad_none(2, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert to_list(array.pad_none(3, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.pad_none(3, 1).form == array.pad_none(3, 1).form
    assert to_list(array.pad_none(4, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None],
    ]
    assert array.typetracer.pad_none(4, 1).form == array.pad_none(4, 1).form
    assert to_list(array.pad_none(5, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None, None],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None, None],
    ]
    assert array.typetracer.pad_none(5, 1).form == array.pad_none(5, 1).form

    assert to_list(array.pad_none(3, 2)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.pad_none(3, 2).form == array.pad_none(3, 2).form

    assert to_list(array.pad_none(7, 2)) == [
        [
            [0, 1, 2, 3, 4, None, None],
            [5, 6, 7, 8, 9, None, None],
            [10, 11, 12, 13, 14, None, None],
        ],
        [
            [15, 16, 17, 18, 19, None, None],
            [20, 21, 22, 23, 24, None, None],
            [25, 26, 27, 28, 29, None, None],
        ],
    ]
    assert array.typetracer.pad_none(7, 2).form == array.pad_none(7, 2).form

    assert to_list(array.pad_none(2, 2)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.pad_none(2, 2).form == array.pad_none(2, 2).form


def test_rpad_and_clip_regular_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array(
            [
                2.1,
                8.4,
                7.4,
                1.6,
                2.2,
                3.4,
                6.2,
                5.4,
                1.5,
                3.9,
                3.8,
                3.0,
                8.5,
                6.9,
                4.3,
                3.6,
                6.7,
                1.8,
                3.2,
            ]
        )
    )
    index = ak._v2.index.Index64(
        np.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=np.int64)
    )
    indexedarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(index, content)
    array = ak._v2.contents.regulararray.RegularArray(indexedarray, 3, zeros_length=0)

    assert to_list(array.pad_none(5, 0, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
        None,
    ]
    assert (
        array.typetracer.pad_none(5, 0, clip=True).form
        == array.pad_none(5, 0, clip=True).form
    )
    assert to_list(array.pad_none(4, 0, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
    ]
    assert (
        array.typetracer.pad_none(4, 0, clip=True).form
        == array.pad_none(4, 0, clip=True).form
    )
    assert to_list(array.pad_none(3, 0, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert (
        array.typetracer.pad_none(3, 0, clip=True).form
        == array.pad_none(3, 0, clip=True).form
    )
    assert to_list(array.pad_none(2, 0, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
    ]
    assert (
        array.typetracer.pad_none(2, 0, clip=True).form
        == array.pad_none(2, 0, clip=True).form
    )
    assert to_list(array.pad_none(1, 0, clip=True)) == [[6.9, 3.9, 6.9]]
    assert (
        array.typetracer.pad_none(1, 0, clip=True).form
        == array.pad_none(1, 0, clip=True).form
    )
    assert to_list(array.pad_none(5, 1, clip=True)) == [
        [6.9, 3.9, 6.9, None, None],
        [2.2, 1.5, 1.6, None, None],
        [3.6, None, 6.7, None, None],
    ]
    assert (
        array.typetracer.pad_none(5, 1, clip=True).form
        == array.pad_none(5, 1, clip=True).form
    )
    assert to_list(array.pad_none(4, 1, clip=True)) == [
        [6.9, 3.9, 6.9, None],
        [2.2, 1.5, 1.6, None],
        [3.6, None, 6.7, None],
    ]
    assert (
        array.typetracer.pad_none(4, 1, clip=True).form
        == array.pad_none(4, 1, clip=True).form
    )
    assert to_list(array.pad_none(3, 1, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert (
        array.typetracer.pad_none(3, 1, clip=True).form
        == array.pad_none(3, 1, clip=True).form
    )
    assert to_list(array.pad_none(2, 1, clip=True)) == [
        [6.9, 3.9],
        [2.2, 1.5],
        [3.6, None],
    ]
    assert (
        array.typetracer.pad_none(2, 1, clip=True).form
        == array.pad_none(2, 1, clip=True).form
    )
    assert to_list(array.pad_none(1, 1, clip=True)) == [[6.9], [2.2], [3.6]]
    assert (
        array.typetracer.pad_none(1, 1, clip=True).form
        == array.pad_none(1, 1, clip=True).form
    )

    array = ak._v2.contents.numpyarray.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert to_list(array) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert to_list(array.pad_none(7, 2, clip=True)) == [
        [
            [0, 1, 2, 3, 4, None, None],
            [5, 6, 7, 8, 9, None, None],
            [10, 11, 12, 13, 14, None, None],
        ],
        [
            [15, 16, 17, 18, 19, None, None],
            [20, 21, 22, 23, 24, None, None],
            [25, 26, 27, 28, 29, None, None],
        ],
    ]
    assert (
        array.typetracer.pad_none(7, 2, clip=True).form
        == array.pad_none(7, 2, clip=True).form
    )

    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    regulararray = ak._v2.contents.regulararray.RegularArray(
        listoffsetarray, 2, zeros_length=0
    )

    assert to_list(regulararray.pad_none(1, 0, clip=True)) == [[[0.0, 1.1, 2.2], []]]
    assert (
        regulararray.typetracer.pad_none(1, 0, clip=True).form
        == regulararray.pad_none(1, 0, clip=True).form
    )
    assert to_list(regulararray.pad_none(2, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
    ]
    assert (
        regulararray.typetracer.pad_none(2, 0, clip=True).form
        == regulararray.pad_none(2, 0, clip=True).form
    )
    assert to_list(regulararray.pad_none(3, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.pad_none(3, 0, clip=True).form
        == regulararray.pad_none(3, 0, clip=True).form
    )
    assert to_list(regulararray.pad_none(4, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
    ]
    assert (
        regulararray.typetracer.pad_none(4, 0, clip=True).form
        == regulararray.pad_none(4, 0, clip=True).form
    )
    assert to_list(regulararray.pad_none(5, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
        None,
    ]
    assert (
        regulararray.typetracer.pad_none(5, 0, clip=True).form
        == regulararray.pad_none(5, 0, clip=True).form
    )

    assert to_list(regulararray.pad_none(1, 1, clip=True)) == [
        [[0.0, 1.1, 2.2]],
        [[3.3, 4.4]],
        [[6.6, 7.7, 8.8, 9.9]],
    ]
    assert (
        regulararray.typetracer.pad_none(1, 1, clip=True).form
        == regulararray.pad_none(1, 1, clip=True).form
    )
    assert to_list(regulararray.pad_none(2, 1, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.pad_none(2, 1, clip=True).form
        == regulararray.pad_none(2, 1, clip=True).form
    )
    assert to_list(regulararray.pad_none(3, 1, clip=True)) == [
        [[0.0, 1.1, 2.2], [], None],
        [[3.3, 4.4], [5.5], None],
        [[6.6, 7.7, 8.8, 9.9], [], None],
    ]
    assert (
        regulararray.typetracer.pad_none(3, 1, clip=True).form
        == regulararray.pad_none(3, 1, clip=True).form
    )
    assert to_list(regulararray.pad_none(7, 1, clip=True)) == [
        [[0.0, 1.1, 2.2], [], None, None, None, None, None],
        [[3.3, 4.4], [5.5], None, None, None, None, None],
        [[6.6, 7.7, 8.8, 9.9], [], None, None, None, None, None],
    ]
    assert (
        regulararray.typetracer.pad_none(7, 1, clip=True).form
        == regulararray.pad_none(7, 1, clip=True).form
    )

    assert to_list(regulararray.pad_none(1, 2, clip=True)) == [
        [[0.0], [None]],
        [[3.3], [5.5]],
        [[6.6], [None]],
    ]
    assert (
        regulararray.typetracer.pad_none(1, 2, clip=True).form
        == regulararray.pad_none(1, 2, clip=True).form
    )
    assert to_list(regulararray.pad_none(2, 2, clip=True)) == [
        [[0.0, 1.1], [None, None]],
        [[3.3, 4.4], [5.5, None]],
        [[6.6, 7.7], [None, None]],
    ]
    assert (
        regulararray.typetracer.pad_none(2, 2, clip=True).form
        == regulararray.pad_none(2, 2, clip=True).form
    )
    assert to_list(regulararray.pad_none(3, 2, clip=True)) == [
        [[0.0, 1.1, 2.2], [None, None, None]],
        [[3.3, 4.4, None], [5.5, None, None]],
        [[6.6, 7.7, 8.8], [None, None, None]],
    ]
    assert (
        regulararray.typetracer.pad_none(3, 2, clip=True).form
        == regulararray.pad_none(3, 2, clip=True).form
    )
    assert to_list(regulararray.pad_none(4, 2, clip=True)) == [
        [[0.0, 1.1, 2.2, None], [None, None, None, None]],
        [[3.3, 4.4, None, None], [5.5, None, None, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None, None, None]],
    ]
    assert (
        regulararray.typetracer.pad_none(4, 2, clip=True).form
        == regulararray.pad_none(4, 2, clip=True).form
    )
    assert to_list(regulararray.pad_none(5, 2, clip=True)) == [
        [[0.0, 1.1, 2.2, None, None], [None, None, None, None, None]],
        [[3.3, 4.4, None, None, None], [5.5, None, None, None, None]],
        [[6.6, 7.7, 8.8, 9.9, None], [None, None, None, None, None]],
    ]
    assert (
        regulararray.typetracer.pad_none(5, 2, clip=True).form
        == regulararray.pad_none(5, 2, clip=True).form
    )


def test_rpad_regular_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array(
            [
                2.1,
                8.4,
                7.4,
                1.6,
                2.2,
                3.4,
                6.2,
                5.4,
                1.5,
                3.9,
                3.8,
                3.0,
                8.5,
                6.9,
                4.3,
                3.6,
                6.7,
                1.8,
                3.2,
            ]
        )
    )
    index = ak._v2.index.Index64(
        np.array([13, 9, 13, 4, 8, 3, 15, -1, 16, 2, 8], dtype=np.int64)
    )
    indexedarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(index, content)
    array = ak._v2.contents.regulararray.RegularArray(indexedarray, 3, zeros_length=0)

    assert to_list(array.pad_none(5, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
        None,
    ]
    assert array.typetracer.pad_none(5, 0).form == array.pad_none(5, 0).form
    assert to_list(array.pad_none(4, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
    ]
    assert array.typetracer.pad_none(4, 0).form == array.pad_none(4, 0).form
    assert to_list(array.pad_none(3, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.pad_none(3, 0).form == array.pad_none(3, 0).form
    assert to_list(array.pad_none(1, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.pad_none(1, 0).form == array.pad_none(1, 0).form
    assert to_list(array.pad_none(5, 1)) == [
        [6.9, 3.9, 6.9, None, None],
        [2.2, 1.5, 1.6, None, None],
        [3.6, None, 6.7, None, None],
    ]
    assert array.typetracer.pad_none(5, 1).form == array.pad_none(5, 1).form
    assert to_list(array.pad_none(4, 1)) == [
        [6.9, 3.9, 6.9, None],
        [2.2, 1.5, 1.6, None],
        [3.6, None, 6.7, None],
    ]
    assert array.typetracer.pad_none(4, 1).form == array.pad_none(4, 1).form
    assert to_list(array.pad_none(3, 1)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.pad_none(3, 1).form == array.pad_none(3, 1).form
    assert to_list(array.pad_none(1, 1)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.pad_none(1, 1).form == array.pad_none(1, 1).form

    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    regulararray = ak._v2.contents.regulararray.RegularArray(
        listoffsetarray, 2, zeros_length=0
    )

    assert to_list(regulararray.pad_none(1, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.pad_none(1, 0).form == regulararray.pad_none(1, 0).form
    )
    assert to_list(regulararray.pad_none(3, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.pad_none(3, 0).form == regulararray.pad_none(3, 0).form
    )
    assert to_list(regulararray.pad_none(4, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
    ]
    assert (
        regulararray.typetracer.pad_none(4, 0).form == regulararray.pad_none(4, 0).form
    )
    assert to_list(regulararray.pad_none(7, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
        None,
        None,
        None,
    ]
    assert (
        regulararray.typetracer.pad_none(7, 0).form == regulararray.pad_none(7, 0).form
    )

    assert to_list(regulararray.pad_none(1, 1)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.pad_none(1, 1).form == regulararray.pad_none(1, 1).form
    )
    assert to_list(regulararray.pad_none(2, 1)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.pad_none(2, 1).form == regulararray.pad_none(2, 1).form
    )
    assert to_list(regulararray.pad_none(3, 1)) == [
        [[0.0, 1.1, 2.2], [], None],
        [[3.3, 4.4], [5.5], None],
        [[6.6, 7.7, 8.8, 9.9], [], None],
    ]
    assert (
        regulararray.typetracer.pad_none(3, 1).form == regulararray.pad_none(3, 1).form
    )
    assert to_list(regulararray.pad_none(5, 1)) == [
        [[0.0, 1.1, 2.2], [], None, None, None],
        [[3.3, 4.4], [5.5], None, None, None],
        [[6.6, 7.7, 8.8, 9.9], [], None, None, None],
    ]
    assert (
        regulararray.typetracer.pad_none(5, 1).form == regulararray.pad_none(5, 1).form
    )
    assert to_list(regulararray.pad_none(7, 1)) == [
        [[0.0, 1.1, 2.2], [], None, None, None, None, None],
        [[3.3, 4.4], [5.5], None, None, None, None, None],
        [[6.6, 7.7, 8.8, 9.9], [], None, None, None, None, None],
    ]
    assert (
        regulararray.typetracer.pad_none(7, 1).form == regulararray.pad_none(7, 1).form
    )

    assert to_list(regulararray.pad_none(1, 2)) == [
        [[0.0, 1.1, 2.2], [None]],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], [None]],
    ]
    assert (
        regulararray.typetracer.pad_none(1, 2).form == regulararray.pad_none(1, 2).form
    )
    assert to_list(regulararray.pad_none(2, 2)) == [
        [[0.0, 1.1, 2.2], [None, None]],
        [[3.3, 4.4], [5.5, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None]],
    ]
    assert (
        regulararray.typetracer.pad_none(2, 2).form == regulararray.pad_none(2, 2).form
    )
    assert to_list(regulararray.pad_none(3, 2)) == [
        [[0.0, 1.1, 2.2], [None, None, None]],
        [[3.3, 4.4, None], [5.5, None, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None, None]],
    ]
    assert (
        regulararray.typetracer.pad_none(3, 2).form == regulararray.pad_none(3, 2).form
    )
    assert to_list(regulararray.pad_none(4, 2)) == [
        [[0.0, 1.1, 2.2, None], [None, None, None, None]],
        [[3.3, 4.4, None, None], [5.5, None, None, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None, None, None]],
    ]
    assert (
        regulararray.typetracer.pad_none(4, 2).form == regulararray.pad_none(4, 2).form
    )


def test_rpad_and_clip_listoffset_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    assert to_list(listoffsetarray) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
    ]

    assert to_list(listoffsetarray.pad_none(3, 0, clip=True)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(3, 0, clip=True).form
        == listoffsetarray.pad_none(3, 0, clip=True).form
    )
    assert "option[" + str(listoffsetarray.form.type) + "]" == str(
        listoffsetarray.pad_none(3, 0, clip=True).form.type
    )

    assert to_list(listoffsetarray.pad_none(7, 0, clip=True)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
        None,
    ]
    assert (
        listoffsetarray.typetracer.pad_none(7, 0, clip=True).form
        == listoffsetarray.pad_none(7, 0, clip=True).form
    )
    assert "option[" + str(listoffsetarray.form.type) + "]" == str(
        listoffsetarray.pad_none(7, 0, clip=True).form.type
    )

    assert to_list(listoffsetarray.pad_none(5, 1, clip=True)) == [
        [0.0, 1.1, 2.2, None, None],
        [None, None, None, None, None],
        [3.3, 4.4, None, None, None],
        [5.5, None, None, None, None],
        [6.6, 7.7, 8.8, 9.9, None],
        [None, None, None, None, None],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(5, 1, clip=True).form
        == listoffsetarray.pad_none(5, 1, clip=True).form
    )

    assert str(listoffsetarray.pad_none(5, 1).form.type) == "var * ?float64"
    assert str(listoffsetarray.pad_none(5, 1, clip=True).form.type) == "5 * ?float64"

    assert to_list(listoffsetarray.pad_none(1, 1, clip=True)) == [
        [0.0],
        [None],
        [3.3],
        [5.5],
        [6.6],
        [None],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(1, 1, clip=True).form
        == listoffsetarray.pad_none(1, 1, clip=True).form
    )

    content = ak._v2.contents.numpyarray.NumpyArray(np.array([1.5, 3.3]))
    index = ak._v2.index.Index64(
        np.array(
            [
                0,
                -3,
                1,
                -2,
                1,
                0,
                0,
                -3,
                -13,
                0,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                -10,
                0,
                -1,
                0,
                0,
                0,
                1,
                -1,
                1,
                1,
            ]
        )
    )
    indexedarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(index, content)
    offsets = ak._v2.index.Index64(np.array([14, 15, 15, 15, 26, 26, 26]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(
        offsets, indexedarray
    )

    assert to_list(listoffsetarray) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]
    assert to_list(listoffsetarray.pad_none(1, 0, clip=True)) == [[3.3]]
    assert (
        listoffsetarray.typetracer.pad_none(1, 0, clip=True).form
        == listoffsetarray.pad_none(1, 0, clip=True).form
    )
    assert to_list(listoffsetarray.pad_none(1, 1, clip=True)) == [
        [3.3],
        [None],
        [None],
        [3.3],
        [None],
        [None],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(1, 1, clip=True).form
        == listoffsetarray.pad_none(1, 1, clip=True).form
    )


def test_rpad_listoffset_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
    ]

    assert to_list(listoffsetarray.pad_none(3, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(3, 0).form
        == listoffsetarray.pad_none(3, 0).form
    )

    assert "option[" + str(ak._v2.operations.type(listoffsetarray)) + "]" == str(
        ak._v2.operations.type(listoffsetarray.pad_none(3, 0))
    )

    assert to_list(listoffsetarray.pad_none(7, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
        None,
    ]
    assert (
        listoffsetarray.typetracer.pad_none(7, 0).form
        == listoffsetarray.pad_none(7, 0).form
    )

    assert "option[" + str(ak._v2.operations.type(listoffsetarray)) + "]" == str(
        ak._v2.operations.type(listoffsetarray.pad_none(7, 0))
    )

    assert to_list(listoffsetarray.pad_none(5, 1)) == [
        [0.0, 1.1, 2.2, None, None],
        [None, None, None, None, None],
        [3.3, 4.4, None, None, None],
        [5.5, None, None, None, None],
        [6.6, 7.7, 8.8, 9.9, None],
        [None, None, None, None, None],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(5, 1).form
        == listoffsetarray.pad_none(5, 1).form
    )
    assert (
        str(ak._v2.operations.type(listoffsetarray.pad_none(5, 1))) == "var * ?float64"
    )

    assert to_list(listoffsetarray.pad_none(1, 1)) == [
        [0.0, 1.1, 2.2],
        [None],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [None],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(1, 1).form
        == listoffsetarray.pad_none(1, 1).form
    )

    content = ak._v2.contents.numpyarray.NumpyArray(np.array([1.5, 3.3]))
    index = ak._v2.index.Index64(
        np.array(
            [
                0,
                -3,
                1,
                -2,
                1,
                0,
                0,
                -3,
                -13,
                0,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                -10,
                0,
                -1,
                0,
                0,
                0,
                1,
                -1,
                1,
                1,
            ]
        )
    )
    indexedarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(index, content)
    offsets = ak._v2.index.Index64(np.array([14, 15, 15, 15, 26, 26, 26]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(
        offsets, indexedarray
    )

    assert to_list(listoffsetarray) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]

    assert to_list(listoffsetarray.pad_none(1, 0)) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(1, 0).form
        == listoffsetarray.pad_none(1, 0).form
    )
    assert f"option[{str(ak._v2.operations.type(listoffsetarray))}]" == str(
        ak._v2.operations.type(listoffsetarray.pad_none(1, 0))
    )

    assert to_list(listoffsetarray.pad_none(6, 0)) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(6, 0).form
        == listoffsetarray.pad_none(6, 0).form
    )
    assert "option[" + str(ak._v2.operations.type(listoffsetarray)) + "]" == str(
        ak._v2.operations.type(listoffsetarray.pad_none(6, 0))
    )

    assert to_list(listoffsetarray.pad_none(7, 0)) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
        None,
    ]
    assert (
        listoffsetarray.typetracer.pad_none(7, 0).form
        == listoffsetarray.pad_none(7, 0).form
    )
    assert "option[" + str(ak._v2.operations.type(listoffsetarray)) + "]" == str(
        ak._v2.operations.type(listoffsetarray.pad_none(7, 0))
    )

    assert to_list(listoffsetarray.pad_none(9, 0)) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
        None,
        None,
        None,
    ]
    assert (
        listoffsetarray.typetracer.pad_none(9, 0).form
        == listoffsetarray.pad_none(9, 0).form
    )
    assert "option[" + str(ak._v2.operations.type(listoffsetarray)) + "]" == str(
        ak._v2.operations.type(listoffsetarray.pad_none(9, 0))
    )

    assert to_list(listoffsetarray.pad_none(1, 1)) == [
        [3.3],
        [None],
        [None],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [None],
        [None],
    ]
    assert (
        listoffsetarray.typetracer.pad_none(1, 1).form
        == listoffsetarray.pad_none(1, 1).form
    )
    assert str(ak._v2.operations.type(listoffsetarray)) == str(
        ak._v2.operations.type(listoffsetarray.pad_none(1, 1))
    )

    assert to_list(listoffsetarray.pad_none(4, 1)) == [
        [3.3, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [None, None, None, None],
        [None, None, None, None],
    ]
    assert str(ak._v2.operations.type(listoffsetarray)) == str(
        ak._v2.operations.type(listoffsetarray.pad_none(4, 1))
    )


def test_rpad_list_array():
    content = ak._v2.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak._v2.index.Index64(np.array([0, 3, 4, 5, 8]))
    stops = ak._v2.index.Index64(np.array([3, 3, 6, 8, 9]))
    array = ak._v2.contents.ListArray(starts, stops, content)

    assert to_list(array) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert to_list(array.pad_none(1, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert f"option[{str(ak._v2.operations.type(array))}]" == str(
        ak._v2.operations.type(array.pad_none(1, 0))
    )

    assert to_list(array.pad_none(2, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert f"option[{str(ak._v2.operations.type(array))}]" == str(
        ak._v2.operations.type(array.pad_none(2, 0))
    )

    assert to_list(array.pad_none(7, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
        None,
        None,
    ]
    assert "option[" + str(ak._v2.operations.type(array)) + "]" == str(
        ak._v2.operations.type(array.pad_none(7, 0))
    )

    assert to_list(array.pad_none(1, 1)) == [
        [0.0, 1.1, 2.2],
        [None],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]

    assert to_list(array.pad_none(2, 1)) == [
        [0.0, 1.1, 2.2],
        [None, None],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8, None],
    ]

    assert to_list(array.pad_none(3, 1)) == [
        [0.0, 1.1, 2.2],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, 6.6, 7.7],
        [8.8, None, None],
    ]

    assert to_list(array.pad_none(4, 1)) == [
        [0.0, 1.1, 2.2, None],
        [None, None, None, None],
        [4.4, 5.5, None, None],
        [5.5, 6.6, 7.7, None],
        [8.8, None, None, None],
    ]


def test_rpad_and_clip_list_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak._v2.index.Index64(np.array([0, 3, 4, 5, 8]))
    stops = ak._v2.index.Index64(np.array([3, 3, 6, 8, 9]))
    array = ak._v2.contents.listarray.ListArray(starts, stops, content)

    assert to_list(array) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert to_list(array.pad_none(1, 0, clip=True)) == [[0.0, 1.1, 2.2]]
    assert (
        array.typetracer.pad_none(1, 0, clip=True).form
        == array.pad_none(1, 0, clip=True).form
    )
    assert "option[" + str(array.form.type) + "]" == str(
        array.pad_none(1, 0, clip=True).form.type
    )

    assert to_list(array.pad_none(2, 0, clip=True)) == [[0.0, 1.1, 2.2], []]
    assert (
        array.typetracer.pad_none(2, 0, clip=True).form
        == array.pad_none(2, 0, clip=True).form
    )
    assert "option[" + str(array.form.type) + "]" == str(
        array.pad_none(2, 0, clip=True).form.type
    )

    assert to_list(array.pad_none(7, 0, clip=True)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
        None,
        None,
    ]
    assert (
        array.typetracer.pad_none(7, 0, clip=True).form
        == array.pad_none(7, 0, clip=True).form
    )
    assert "option[" + str(array.form.type) + "]" == str(
        array.pad_none(7, 0, clip=True).form.type
    )

    assert to_list(array.pad_none(1, 1, clip=True)) == [
        [0.0],
        [None],
        [4.4],
        [5.5],
        [8.8],
    ]
    assert (
        array.typetracer.pad_none(1, 1, clip=True).form
        == array.pad_none(1, 1, clip=True).form
    )

    assert to_list(array.pad_none(2, 1, clip=True)) == [
        [0.0, 1.1],
        [None, None],
        [4.4, 5.5],
        [5.5, 6.6],
        [8.8, None],
    ]
    assert (
        array.typetracer.pad_none(2, 1, clip=True).form
        == array.pad_none(2, 1, clip=True).form
    )


def test_rpad_indexed_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak._v2.index.Index64(np.array([3, 3, 5, 6, 10]))
    listoffsetarray = ak._v2.contents.listarray.ListArray(starts, stops, content)
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([6.6, 7.7, 8.8, 9.9, 5.5, 3.3, 4.4, 0.0, 1.1, 2.2])
    )
    offsets = ak._v2.index.Index64(np.array([0, 4, 5, 7, 7, 10]))
    backward = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)

    index = ak._v2.index.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak._v2.contents.indexedarray.IndexedArray(index, listoffsetarray)
    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]

    assert to_list(backward.pad_none(4, 1)) == to_list(indexedarray.pad_none(4, 1))
    assert to_list(indexedarray.pad_none(1, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert to_list(indexedarray.pad_none(2, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None],
        [3.3, 4.4],
        [None, None],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.pad_none(2, 1).form == indexedarray.pad_none(2, 1).form
    )
    assert to_list(indexedarray.pad_none(3, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None, None],
        [3.3, 4.4, None],
        [None, None, None],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.pad_none(3, 1).form == indexedarray.pad_none(3, 1).form
    )
    assert to_list(indexedarray.pad_none(4, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.pad_none(4, 0).form == indexedarray.pad_none(4, 0).form
    )
    assert to_list(indexedarray.pad_none(5, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.pad_none(5, 0).form == indexedarray.pad_none(5, 0).form
    )
    assert to_list(indexedarray.pad_none(6, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(6, 0).form == indexedarray.pad_none(6, 0).form
    )
    assert to_list(indexedarray.pad_none(7, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(7, 0).form == indexedarray.pad_none(7, 0).form
    )


def test_rpad_and_clip_indexed_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([6.6, 7.7, 8.8, 9.9, 5.5, 3.3, 4.4, 0.0, 1.1, 2.2])
    )
    offsets = ak._v2.index.Index64(np.array([0, 4, 5, 7, 7, 10]))
    backward = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)

    index = ak._v2.index.Index64(np.array([4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak._v2.contents.IndexedArray(index, listoffsetarray)
    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]

    assert to_list(backward.pad_none(4, 1, clip=True)) == to_list(
        indexedarray.pad_none(4, 1, clip=True)
    )
    assert to_list(indexedarray.pad_none(1, 0, clip=True)) == [[6.6, 7.7, 8.8, 9.9]]
    assert (
        indexedarray.typetracer.pad_none(1, 0, clip=True).form
        == indexedarray.pad_none(1, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(2, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
    ]
    assert (
        indexedarray.typetracer.pad_none(2, 0, clip=True).form
        == indexedarray.pad_none(2, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(3, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
    ]
    assert (
        indexedarray.typetracer.pad_none(3, 0, clip=True).form
        == indexedarray.pad_none(3, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(4, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
    ]
    assert (
        indexedarray.typetracer.pad_none(4, 0, clip=True).form
        == indexedarray.pad_none(4, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(5, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.pad_none(5, 0, clip=True).form
        == indexedarray.pad_none(5, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(6, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(6, 0, clip=True).form
        == indexedarray.pad_none(6, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(7, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(7, 0, clip=True).form
        == indexedarray.pad_none(7, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(8, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
        None,
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(8, 0, clip=True).form
        == indexedarray.pad_none(8, 0, clip=True).form
    )

    assert to_list(indexedarray.pad_none(1, 1, clip=True)) == [
        [6.6],
        [5.5],
        [3.3],
        [None],
        [0.0],
    ]
    assert (
        indexedarray.typetracer.pad_none(1, 1, clip=True).form
        == indexedarray.pad_none(1, 1, clip=True).form
    )
    assert to_list(indexedarray.pad_none(2, 1, clip=True)) == [
        [6.6, 7.7],
        [5.5, None],
        [3.3, 4.4],
        [None, None],
        [0.0, 1.1],
    ]
    assert (
        indexedarray.typetracer.pad_none(2, 1, clip=True).form
        == indexedarray.pad_none(2, 1, clip=True).form
    )
    assert to_list(indexedarray.pad_none(3, 1, clip=True)) == [
        [6.6, 7.7, 8.8],
        [5.5, None, None],
        [3.3, 4.4, None],
        [None, None, None],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.pad_none(3, 1, clip=True).form
        == indexedarray.pad_none(3, 1, clip=True).form
    )
    assert to_list(indexedarray.pad_none(4, 1, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None, None, None],
        [3.3, 4.4, None, None],
        [None, None, None, None],
        [0.0, 1.1, 2.2, None],
    ]
    assert (
        indexedarray.typetracer.pad_none(4, 1, clip=True).form
        == indexedarray.pad_none(4, 1, clip=True).form
    )
    assert to_list(indexedarray.pad_none(5, 1, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9, None],
        [5.5, None, None, None, None],
        [3.3, 4.4, None, None, None],
        [None, None, None, None, None],
        [0.0, 1.1, 2.2, None, None],
    ]
    assert (
        indexedarray.typetracer.pad_none(5, 1, clip=True).form
        == indexedarray.pad_none(5, 1, clip=True).form
    )


def test_rpad_indexed_option_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.asarray([0.0, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index = ak._v2.index.Index64(np.asarray([0, -1, -1, 1, 2, 3, 4, 5, 6, 7]))
    offsets = ak._v2.index.Index64(np.asarray([0, 3, 5, 6, 10]))
    indexedarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(index, content)
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(
        offsets, indexedarray
    )
    index = ak._v2.index.Index64(
        np.asarray(
            [
                0,
                -1,
                1,
                2,
                3,
            ]
        )
    )
    listoffsetarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        index, listoffsetarray
    )
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.asarray([6.6, 7.7, 8.8, 9.9, 5.5, 3.3, 4.4, 0.0])
    )
    index = ak._v2.index.Index64(np.asarray([0, 1, 2, 3, 4, 5, 6, 7, -1, -1]))
    offsets = ak._v2.index.Index64(np.asarray([0, 4, 5, 7, 10]))
    indexedarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(index, content)
    listoffsetarray_ = ak._v2.contents.listoffsetarray.ListOffsetArray(
        offsets, indexedarray
    )
    index = ak._v2.index.Index64(
        np.asarray(
            [
                0,
                1,
                2,
                -1,
                3,
            ]
        )
    )
    backward = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        index, listoffsetarray_
    )

    index = ak._v2.index.Index64(np.array([4, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        index, listoffsetarray
    )
    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]

    assert to_list(backward.pad_none(4, 1)) == to_list(indexedarray.pad_none(4, 1))
    assert to_list(indexedarray.pad_none(1, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert (
        indexedarray.typetracer.pad_none(1, 0).form == indexedarray.pad_none(1, 0).form
    )
    assert to_list(indexedarray.pad_none(1, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert (
        indexedarray.typetracer.pad_none(1, 1).form == indexedarray.pad_none(1, 1).form
    )
    assert to_list(indexedarray.pad_none(3, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None, None],
        [3.3, 4.4, None],
        None,
        [0.0, None, None],
    ]
    assert (
        indexedarray.typetracer.pad_none(3, 1).form == indexedarray.pad_none(3, 1).form
    )
    assert to_list(indexedarray.pad_none(4, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert (
        indexedarray.typetracer.pad_none(4, 0).form == indexedarray.pad_none(4, 0).form
    )
    assert to_list(indexedarray.pad_none(5, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert (
        indexedarray.typetracer.pad_none(5, 0).form == indexedarray.pad_none(5, 0).form
    )
    assert to_list(indexedarray.pad_none(6, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(6, 0).form == indexedarray.pad_none(6, 0).form
    )
    assert to_list(indexedarray.pad_none(7, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
        None,
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(7, 0).form == indexedarray.pad_none(7, 0).form
    )
    assert to_list(indexedarray.pad_none(8, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
        None,
        None,
        None,
    ]
    assert (
        indexedarray.typetracer.pad_none(8, 0).form == indexedarray.pad_none(8, 0).form
    )

    assert to_list(indexedarray.pad_none(1, 0, clip=True)) == [[6.6, 7.7, 8.8, 9.9]]
    assert (
        indexedarray.typetracer.pad_none(1, 0, clip=True).form
        == indexedarray.pad_none(1, 0, clip=True).form
    )
    assert to_list(indexedarray.pad_none(1, 1, clip=True)) == [
        [6.6],
        [5.5],
        [3.3],
        None,
        [0.0],
    ]
    assert (
        indexedarray.typetracer.pad_none(1, 1, clip=True).form
        == indexedarray.pad_none(1, 1, clip=True).form
    )


def test_rpad_recordarray():
    keys = ["x", "y"]
    offsets = ak._v2.index.Index64(np.asarray([0, 0, 1, 3]))
    content = ak._v2.contents.numpyarray.NumpyArray(np.asarray([1.1, 2.2, 2.2]))
    content1 = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    offsets = ak._v2.index.Index64(np.asarray([0, 2, 3, 3]))
    content = ak._v2.contents.numpyarray.NumpyArray(np.asarray([2, 2, 1]))
    content2 = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    contents = [content1, content2]
    array = ak._v2.contents.recordarray.RecordArray(contents, keys)

    assert to_list(array.pad_none(5, 0)) == [
        {"x": [], "y": [2, 2]},
        {"x": [1.1], "y": [1]},
        {"x": [2.2, 2.2], "y": []},
        None,
        None,
    ]
    assert array.typetracer.pad_none(5, 0).form == array.pad_none(5, 0).form

    assert to_list(array.pad_none(2, 1)) == [
        {"x": [None, None], "y": [2, 2]},
        {"x": [1.1, None], "y": [1, None]},
        {"x": [2.2, 2.2], "y": [None, None]},
    ]
    assert array.typetracer.pad_none(2, 1).form == array.pad_none(2, 1).form


def test_rpad_unionarray():
    offsets = ak._v2.index.Index64(np.asarray([0, 0, 1, 3]))
    content = ak._v2.contents.numpyarray.NumpyArray(np.asarray([1.1, 2.2, 2.2]))
    content1 = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    offsets = ak._v2.index.Index64(np.asarray([0, 2, 3, 3]))
    content = ak._v2.contents.numpyarray.NumpyArray(np.asarray([2, 2, 1]))
    content2 = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    tags = ak._v2.index.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak._v2.index.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak._v2.contents.unionarray.UnionArray(tags, index, [content1, content2])
    assert to_list(array) == [[], [2, 2], [1.1], [1], [2.2, 2.2], []]

    assert to_list(array.pad_none(7, 0)) == [
        [],
        [2, 2],
        [1.1],
        [1],
        [2.2, 2.2],
        [],
        None,
    ]
    assert array.typetracer.pad_none(7, 0).form == array.pad_none(7, 0).form

    assert to_list(array.pad_none(2, 1)) == [
        [None, None],
        [2, 2],
        [1.1, None],
        [1, None],
        [2.2, 2.2],
        [None, None],
    ]
    assert array.typetracer.pad_none(2, 1).form == array.pad_none(2, 1).form
