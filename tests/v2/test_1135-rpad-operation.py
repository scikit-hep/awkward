# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401
from awkward._v2.tmp_for_testing import v1_to_v2  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_rpad_and_clip_empty_array():
    empty = ak._v2.contents.emptyarray.EmptyArray()
    assert ak.to_list(empty) == []
    assert ak.to_list(empty.rpad(5, 0)) == [None, None, None, None, None]
    assert empty.typetracer.rpad(5, 0).form == empty.rpad(5, 0).form
    assert ak.to_list(empty.rpad(5, 0, clip=True)) == [None, None, None, None, None]
    assert (
        empty.typetracer.rpad(5, 0, clip=True).form == empty.rpad(5, 0, clip=True).form
    )


def test_rpad_and_clip_numpy_array():
    array = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    assert ak.to_list(array) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert ak.to_list(array.rpad(5, 0, clip=True)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
        None,
        None,
    ]
    assert (
        array.typetracer.rpad(5, 0, clip=True).form == array.rpad(5, 0, clip=True).form
    )

    assert ak.to_list(array.rpad(5, 1, clip=True)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None, None],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None, None],
    ]
    assert (
        array.typetracer.rpad(5, 1, clip=True).form == array.rpad(5, 1, clip=True).form
    )

    assert ak.to_list(array.rpad(7, 2, clip=True)) == [
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
        array.typetracer.rpad(7, 2, clip=True).form == array.rpad(7, 2, clip=True).form
    )

    assert ak.to_list(array.rpad(2, 2, clip=True)) == [
        [[0, 1], [5, 6], [10, 11]],
        [[15, 16], [20, 21], [25, 26]],
    ]
    assert (
        array.typetracer.rpad(2, 2, clip=True).form == array.rpad(2, 2, clip=True).form
    )


def test_rpad_numpy_array():
    array = ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    assert ak.to_list(array.rpad(10, 0)) == [
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
    assert array.typetracer.rpad(10, 0).form == array.rpad(10, 0).form

    array = ak._v2.contents.numpyarray.NumpyArray(
        np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    )
    assert ak.to_list(array.rpad(5, 0)) == [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        None,
        None,
        None,
    ]
    assert array.typetracer.rpad(5, 0).form == array.rpad(5, 0).form
    assert ak.to_list(array.rpad(5, 1)) == [
        [1.1, 2.2, 3.3, None, None],
        [4.4, 5.5, 6.6, None, None],
    ]
    assert array.typetracer.rpad(5, 1).form == array.rpad(5, 1).form

    array = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    assert ak.to_list(array) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert ak.to_list(array.rpad(1, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.rpad(1, 0).form == array.rpad(1, 0).form
    assert ak.to_list(array.rpad(2, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.rpad(2, 0).form == array.rpad(2, 0).form
    assert ak.to_list(array.rpad(3, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
    ]
    assert array.typetracer.rpad(3, 0).form == array.rpad(3, 0).form
    assert ak.to_list(array.rpad(4, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
        None,
    ]
    assert array.typetracer.rpad(4, 0).form == array.rpad(4, 0).form
    assert ak.to_list(array.rpad(5, 0)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        None,
        None,
        None,
    ]
    assert array.typetracer.rpad(5, 0).form == array.rpad(5, 0).form

    assert ak.to_list(array.rpad(2, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert ak.to_list(array.rpad(3, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.rpad(3, 1).form == array.rpad(3, 1).form
    assert ak.to_list(array.rpad(4, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None],
    ]
    assert array.typetracer.rpad(4, 1).form == array.rpad(4, 1).form
    assert ak.to_list(array.rpad(5, 1)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], None, None],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29], None, None],
    ]
    assert array.typetracer.rpad(5, 1).form == array.rpad(5, 1).form

    assert ak.to_list(array.rpad(3, 2)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.rpad(3, 2).form == array.rpad(3, 2).form

    assert ak.to_list(array.rpad(7, 2)) == [
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
    assert array.typetracer.rpad(7, 2).form == array.rpad(7, 2).form

    assert ak.to_list(array.rpad(2, 2)) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert array.typetracer.rpad(2, 2).form == array.rpad(2, 2).form


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

    assert ak.to_list(array.rpad(5, 0, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
        None,
    ]
    assert (
        array.typetracer.rpad(5, 0, clip=True).form == array.rpad(5, 0, clip=True).form
    )
    assert ak.to_list(array.rpad(4, 0, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
    ]
    assert (
        array.typetracer.rpad(4, 0, clip=True).form == array.rpad(4, 0, clip=True).form
    )
    assert ak.to_list(array.rpad(3, 0, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert (
        array.typetracer.rpad(3, 0, clip=True).form == array.rpad(3, 0, clip=True).form
    )
    assert ak.to_list(array.rpad(2, 0, clip=True)) == [[6.9, 3.9, 6.9], [2.2, 1.5, 1.6]]
    assert (
        array.typetracer.rpad(2, 0, clip=True).form == array.rpad(2, 0, clip=True).form
    )
    assert ak.to_list(array.rpad(1, 0, clip=True)) == [[6.9, 3.9, 6.9]]
    assert (
        array.typetracer.rpad(1, 0, clip=True).form == array.rpad(1, 0, clip=True).form
    )
    assert ak.to_list(array.rpad(5, 1, clip=True)) == [
        [6.9, 3.9, 6.9, None, None],
        [2.2, 1.5, 1.6, None, None],
        [3.6, None, 6.7, None, None],
    ]
    assert (
        array.typetracer.rpad(5, 1, clip=True).form == array.rpad(5, 1, clip=True).form
    )
    assert ak.to_list(array.rpad(4, 1, clip=True)) == [
        [6.9, 3.9, 6.9, None],
        [2.2, 1.5, 1.6, None],
        [3.6, None, 6.7, None],
    ]
    assert (
        array.typetracer.rpad(4, 1, clip=True).form == array.rpad(4, 1, clip=True).form
    )
    assert ak.to_list(array.rpad(3, 1, clip=True)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert (
        array.typetracer.rpad(3, 1, clip=True).form == array.rpad(3, 1, clip=True).form
    )
    assert ak.to_list(array.rpad(2, 1, clip=True)) == [
        [6.9, 3.9],
        [2.2, 1.5],
        [3.6, None],
    ]
    assert (
        array.typetracer.rpad(2, 1, clip=True).form == array.rpad(2, 1, clip=True).form
    )
    assert ak.to_list(array.rpad(1, 1, clip=True)) == [[6.9], [2.2], [3.6]]
    assert (
        array.typetracer.rpad(1, 1, clip=True).form == array.rpad(1, 1, clip=True).form
    )

    array = ak._v2.contents.numpyarray.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5))
    assert ak.to_list(array) == [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]

    assert ak.to_list(array.rpad(7, 2, clip=True)) == [
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
        array.typetracer.rpad(7, 2, clip=True).form == array.rpad(7, 2, clip=True).form
    )

    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    regulararray = ak._v2.contents.regulararray.RegularArray(
        listoffsetarray, 2, zeros_length=0
    )

    assert ak.to_list(regulararray.rpad(1, 0, clip=True)) == [[[0.0, 1.1, 2.2], []]]
    assert (
        regulararray.typetracer.rpad(1, 0, clip=True).form
        == regulararray.rpad(1, 0, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(2, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
    ]
    assert (
        regulararray.typetracer.rpad(2, 0, clip=True).form
        == regulararray.rpad(2, 0, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(3, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.rpad(3, 0, clip=True).form
        == regulararray.rpad(3, 0, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(4, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
    ]
    assert (
        regulararray.typetracer.rpad(4, 0, clip=True).form
        == regulararray.rpad(4, 0, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(5, 0, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
        None,
    ]
    assert (
        regulararray.typetracer.rpad(5, 0, clip=True).form
        == regulararray.rpad(5, 0, clip=True).form
    )

    assert ak.to_list(regulararray.rpad(1, 1, clip=True)) == [
        [[0.0, 1.1, 2.2]],
        [[3.3, 4.4]],
        [[6.6, 7.7, 8.8, 9.9]],
    ]
    assert (
        regulararray.typetracer.rpad(1, 1, clip=True).form
        == regulararray.rpad(1, 1, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(2, 1, clip=True)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert (
        regulararray.typetracer.rpad(2, 1, clip=True).form
        == regulararray.rpad(2, 1, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(3, 1, clip=True)) == [
        [[0.0, 1.1, 2.2], [], None],
        [[3.3, 4.4], [5.5], None],
        [[6.6, 7.7, 8.8, 9.9], [], None],
    ]
    assert (
        regulararray.typetracer.rpad(3, 1, clip=True).form
        == regulararray.rpad(3, 1, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(7, 1, clip=True)) == [
        [[0.0, 1.1, 2.2], [], None, None, None, None, None],
        [[3.3, 4.4], [5.5], None, None, None, None, None],
        [[6.6, 7.7, 8.8, 9.9], [], None, None, None, None, None],
    ]
    assert (
        regulararray.typetracer.rpad(7, 1, clip=True).form
        == regulararray.rpad(7, 1, clip=True).form
    )

    assert ak.to_list(regulararray.rpad(1, 2, clip=True)) == [
        [[0.0], [None]],
        [[3.3], [5.5]],
        [[6.6], [None]],
    ]
    assert (
        regulararray.typetracer.rpad(1, 2, clip=True).form
        == regulararray.rpad(1, 2, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(2, 2, clip=True)) == [
        [[0.0, 1.1], [None, None]],
        [[3.3, 4.4], [5.5, None]],
        [[6.6, 7.7], [None, None]],
    ]
    assert (
        regulararray.typetracer.rpad(2, 2, clip=True).form
        == regulararray.rpad(2, 2, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(3, 2, clip=True)) == [
        [[0.0, 1.1, 2.2], [None, None, None]],
        [[3.3, 4.4, None], [5.5, None, None]],
        [[6.6, 7.7, 8.8], [None, None, None]],
    ]
    assert (
        regulararray.typetracer.rpad(3, 2, clip=True).form
        == regulararray.rpad(3, 2, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(4, 2, clip=True)) == [
        [[0.0, 1.1, 2.2, None], [None, None, None, None]],
        [[3.3, 4.4, None, None], [5.5, None, None, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None, None, None]],
    ]
    assert (
        regulararray.typetracer.rpad(4, 2, clip=True).form
        == regulararray.rpad(4, 2, clip=True).form
    )
    assert ak.to_list(regulararray.rpad(5, 2, clip=True)) == [
        [[0.0, 1.1, 2.2, None, None], [None, None, None, None, None]],
        [[3.3, 4.4, None, None, None], [5.5, None, None, None, None]],
        [[6.6, 7.7, 8.8, 9.9, None], [None, None, None, None, None]],
    ]
    assert (
        regulararray.typetracer.rpad(5, 2, clip=True).form
        == regulararray.rpad(5, 2, clip=True).form
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

    assert ak.to_list(array.rpad(5, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
        None,
    ]
    assert array.typetracer.rpad(5, 0).form == array.rpad(5, 0).form
    assert ak.to_list(array.rpad(4, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
        None,
    ]
    assert array.typetracer.rpad(4, 0).form == array.rpad(4, 0).form
    assert ak.to_list(array.rpad(3, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.rpad(3, 0).form == array.rpad(3, 0).form
    assert ak.to_list(array.rpad(1, 0)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.rpad(1, 0).form == array.rpad(1, 0).form
    assert ak.to_list(array.rpad(5, 1)) == [
        [6.9, 3.9, 6.9, None, None],
        [2.2, 1.5, 1.6, None, None],
        [3.6, None, 6.7, None, None],
    ]
    assert array.typetracer.rpad(5, 1).form == array.rpad(5, 1).form
    assert ak.to_list(array.rpad(4, 1)) == [
        [6.9, 3.9, 6.9, None],
        [2.2, 1.5, 1.6, None],
        [3.6, None, 6.7, None],
    ]
    assert array.typetracer.rpad(4, 1).form == array.rpad(4, 1).form
    assert ak.to_list(array.rpad(3, 1)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.rpad(3, 1).form == array.rpad(3, 1).form
    assert ak.to_list(array.rpad(1, 1)) == [
        [6.9, 3.9, 6.9],
        [2.2, 1.5, 1.6],
        [3.6, None, 6.7],
    ]
    assert array.typetracer.rpad(1, 1).form == array.rpad(1, 1).form

    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    regulararray = ak._v2.contents.regulararray.RegularArray(
        listoffsetarray, 2, zeros_length=0
    )

    assert ak.to_list(regulararray.rpad(1, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert regulararray.typetracer.rpad(1, 0).form == regulararray.rpad(1, 0).form
    assert ak.to_list(regulararray.rpad(3, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert regulararray.typetracer.rpad(3, 0).form == regulararray.rpad(3, 0).form
    assert ak.to_list(regulararray.rpad(4, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
    ]
    assert regulararray.typetracer.rpad(4, 0).form == regulararray.rpad(4, 0).form
    assert ak.to_list(regulararray.rpad(7, 0)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
        None,
        None,
        None,
        None,
    ]
    assert regulararray.typetracer.rpad(7, 0).form == regulararray.rpad(7, 0).form

    assert ak.to_list(regulararray.rpad(1, 1)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert regulararray.typetracer.rpad(1, 1).form == regulararray.rpad(1, 1).form
    assert ak.to_list(regulararray.rpad(2, 1)) == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert regulararray.typetracer.rpad(2, 1).form == regulararray.rpad(2, 1).form
    assert ak.to_list(regulararray.rpad(3, 1)) == [
        [[0.0, 1.1, 2.2], [], None],
        [[3.3, 4.4], [5.5], None],
        [[6.6, 7.7, 8.8, 9.9], [], None],
    ]
    assert regulararray.typetracer.rpad(3, 1).form == regulararray.rpad(3, 1).form
    assert ak.to_list(regulararray.rpad(5, 1)) == [
        [[0.0, 1.1, 2.2], [], None, None, None],
        [[3.3, 4.4], [5.5], None, None, None],
        [[6.6, 7.7, 8.8, 9.9], [], None, None, None],
    ]
    assert regulararray.typetracer.rpad(5, 1).form == regulararray.rpad(5, 1).form
    assert ak.to_list(regulararray.rpad(7, 1)) == [
        [[0.0, 1.1, 2.2], [], None, None, None, None, None],
        [[3.3, 4.4], [5.5], None, None, None, None, None],
        [[6.6, 7.7, 8.8, 9.9], [], None, None, None, None, None],
    ]
    assert regulararray.typetracer.rpad(7, 1).form == regulararray.rpad(7, 1).form

    assert ak.to_list(regulararray.rpad(1, 2)) == [
        [[0.0, 1.1, 2.2], [None]],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], [None]],
    ]
    assert regulararray.typetracer.rpad(1, 2).form == regulararray.rpad(1, 2).form
    assert ak.to_list(regulararray.rpad(2, 2)) == [
        [[0.0, 1.1, 2.2], [None, None]],
        [[3.3, 4.4], [5.5, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None]],
    ]
    assert regulararray.typetracer.rpad(2, 2).form == regulararray.rpad(2, 2).form
    assert ak.to_list(regulararray.rpad(3, 2)) == [
        [[0.0, 1.1, 2.2], [None, None, None]],
        [[3.3, 4.4, None], [5.5, None, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None, None]],
    ]
    assert regulararray.typetracer.rpad(3, 2).form == regulararray.rpad(3, 2).form
    assert ak.to_list(regulararray.rpad(4, 2)) == [
        [[0.0, 1.1, 2.2, None], [None, None, None, None]],
        [[3.3, 4.4, None, None], [5.5, None, None, None]],
        [[6.6, 7.7, 8.8, 9.9], [None, None, None, None]],
    ]
    assert regulararray.typetracer.rpad(4, 2).form == regulararray.rpad(4, 2).form


def test_rpad_and_clip_listoffset_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)
    assert ak.to_list(listoffsetarray) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
    ]

    assert ak.to_list(listoffsetarray.rpad(3, 0, clip=True)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
    ]
    assert (
        listoffsetarray.typetracer.rpad(3, 0, clip=True).form
        == listoffsetarray.rpad(3, 0, clip=True).form
    )
    assert str("option[") + str(listoffsetarray.form.type) + str("]") == str(
        listoffsetarray.rpad(3, 0, clip=True).form.type
    )

    assert ak.to_list(listoffsetarray.rpad(7, 0, clip=True)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
        None,
    ]
    assert (
        listoffsetarray.typetracer.rpad(7, 0, clip=True).form
        == listoffsetarray.rpad(7, 0, clip=True).form
    )
    assert str("option[") + str(listoffsetarray.form.type) + str("]") == str(
        listoffsetarray.rpad(7, 0, clip=True).form.type
    )

    assert ak.to_list(listoffsetarray.rpad(5, 1, clip=True)) == [
        [0.0, 1.1, 2.2, None, None],
        [None, None, None, None, None],
        [3.3, 4.4, None, None, None],
        [5.5, None, None, None, None],
        [6.6, 7.7, 8.8, 9.9, None],
        [None, None, None, None, None],
    ]
    assert (
        listoffsetarray.typetracer.rpad(5, 1, clip=True).form
        == listoffsetarray.rpad(5, 1, clip=True).form
    )

    assert str(listoffsetarray.rpad(5, 1).form.type) == "var * ?float64"
    assert str(listoffsetarray.rpad(5, 1, clip=True).form.type) == "5 * ?float64"

    assert ak.to_list(listoffsetarray.rpad(1, 1, clip=True)) == [
        [0.0],
        [None],
        [3.3],
        [5.5],
        [6.6],
        [None],
    ]
    assert (
        listoffsetarray.typetracer.rpad(1, 1, clip=True).form
        == listoffsetarray.rpad(1, 1, clip=True).form
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

    assert ak.to_list(listoffsetarray) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]
    assert ak.to_list(listoffsetarray.rpad(1, 0, clip=True)) == [[3.3]]
    assert (
        listoffsetarray.typetracer.rpad(1, 0, clip=True).form
        == listoffsetarray.rpad(1, 0, clip=True).form
    )
    assert ak.to_list(listoffsetarray.rpad(1, 1, clip=True)) == [
        [3.3],
        [None],
        [None],
        [3.3],
        [None],
        [None],
    ]
    assert (
        listoffsetarray.typetracer.rpad(1, 1, clip=True).form
        == listoffsetarray.rpad(1, 1, clip=True).form
    )


def test_rpad_listoffset_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak._v2.contents.listoffsetarray.ListOffsetArray(offsets, content)

    assert ak.to_list(listoffsetarray) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
    ]

    assert ak.to_list(listoffsetarray.rpad(3, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
    ]
    assert listoffsetarray.typetracer.rpad(3, 0).form == listoffsetarray.rpad(3, 0).form

    assert str(listoffsetarray.form.type) == str(listoffsetarray.rpad(3, 0).form.type)

    assert ak.to_list(listoffsetarray.rpad(7, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [],
        None,
    ]
    assert listoffsetarray.typetracer.rpad(7, 0).form == listoffsetarray.rpad(7, 0).form
    assert str("option[") + str(listoffsetarray.form.type) + str("]") == str(
        listoffsetarray.rpad(7, 0).form.type
    )

    assert ak.to_list(listoffsetarray.rpad(5, 1)) == [
        [0.0, 1.1, 2.2, None, None],
        [None, None, None, None, None],
        [3.3, 4.4, None, None, None],
        [5.5, None, None, None, None],
        [6.6, 7.7, 8.8, 9.9, None],
        [None, None, None, None, None],
    ]
    assert listoffsetarray.typetracer.rpad(5, 1).form == listoffsetarray.rpad(5, 1).form

    assert ak.to_list(listoffsetarray.rpad(1, 1)) == [
        [0.0, 1.1, 2.2],
        [None],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
        [None],
    ]
    assert listoffsetarray.typetracer.rpad(1, 1).form == listoffsetarray.rpad(1, 1).form

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

    assert ak.to_list(listoffsetarray) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]

    assert ak.to_list(listoffsetarray.rpad(1, 0)) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]
    assert listoffsetarray.typetracer.rpad(1, 0).form == listoffsetarray.rpad(1, 0).form
    assert str(listoffsetarray.form.type) == str(listoffsetarray.rpad(1, 0).form.type)

    assert ak.to_list(listoffsetarray.rpad(6, 0)) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
    ]
    assert listoffsetarray.typetracer.rpad(6, 0).form == listoffsetarray.rpad(6, 0).form
    assert str("option[") + str(listoffsetarray.form.type) + str("]") == str(
        listoffsetarray.rpad(6, 0).form.type
    )

    assert ak.to_list(listoffsetarray.rpad(7, 0)) == [
        [3.3],
        [],
        [],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [],
        [],
        None,
    ]
    assert listoffsetarray.typetracer.rpad(7, 0).form == listoffsetarray.rpad(7, 0).form
    assert str("option[") + str(listoffsetarray.form.type) + str("]") == str(
        listoffsetarray.rpad(7, 0).form.type
    )

    assert ak.to_list(listoffsetarray.rpad(9, 0)) == [
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
    assert listoffsetarray.typetracer.rpad(9, 0).form == listoffsetarray.rpad(9, 0).form
    assert str("option[") + str(listoffsetarray.form.type) + str("]") == str(
        listoffsetarray.rpad(9, 0).form.type
    )

    assert ak.to_list(listoffsetarray.rpad(1, 1)) == [
        [3.3],
        [None],
        [None],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [None],
        [None],
    ]
    assert listoffsetarray.typetracer.rpad(1, 1).form == listoffsetarray.rpad(1, 1).form
    assert str(listoffsetarray.form.type) == str(listoffsetarray.rpad(1, 1).form.type)

    assert ak.to_list(listoffsetarray.rpad(4, 1)) == [
        [3.3, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
        [3.3, 3.3, None, 1.5, None, 1.5, 1.5, 1.5, 3.3, None, 3.3],
        [None, None, None, None],
        [None, None, None, None],
    ]
    assert listoffsetarray.typetracer.rpad(4, 1).form == listoffsetarray.rpad(4, 1).form
    assert str(listoffsetarray.form.type) == str(listoffsetarray.rpad(4, 1).form.type)


def test_rpad_list_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak._v2.index.Index64(np.array([0, 3, 4, 5, 8]))
    stops = ak._v2.index.Index64(np.array([3, 3, 6, 8, 9]))
    array = ak._v2.contents.listarray.ListArray(starts, stops, content)

    assert ak.to_list(array) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert ak.to_list(array.rpad(1, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert str(array.form.type) == str(array.rpad(1, 0).form.type)

    assert ak.to_list(array.rpad(2, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert array.typetracer.rpad(2, 0).form == array.rpad(2, 0).form
    assert str(array.form.type) == str(array.rpad(2, 0).form.type)

    assert ak.to_list(array.rpad(7, 0)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
        None,
        None,
    ]
    assert array.typetracer.rpad(7, 0).form == array.rpad(7, 0).form
    assert str("option[") + str(array.form.type) + str("]") == str(
        array.rpad(7, 0).form.type
    )

    assert ak.to_list(array.rpad(1, 1)) == [
        [0.0, 1.1, 2.2],
        [None],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert array.typetracer.rpad(1, 1).form == array.rpad(1, 1).form

    assert ak.to_list(array.rpad(2, 1)) == [
        [0.0, 1.1, 2.2],
        [None, None],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8, None],
    ]
    assert array.typetracer.rpad(2, 1).form == array.rpad(2, 1).form

    assert ak.to_list(array.rpad(3, 1)) == [
        [0.0, 1.1, 2.2],
        [None, None, None],
        [4.4, 5.5, None],
        [5.5, 6.6, 7.7],
        [8.8, None, None],
    ]
    assert array.typetracer.rpad(3, 1).form == array.rpad(3, 1).form

    assert ak.to_list(array.rpad(4, 1)) == [
        [0.0, 1.1, 2.2, None],
        [None, None, None, None],
        [4.4, 5.5, None, None],
        [5.5, 6.6, 7.7, None],
        [8.8, None, None, None],
    ]
    assert array.typetracer.rpad(4, 1).form == array.rpad(4, 1).form


def test_rpad_and_clip_list_array():
    content = ak._v2.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak._v2.index.Index64(np.array([0, 3, 4, 5, 8]))
    stops = ak._v2.index.Index64(np.array([3, 3, 6, 8, 9]))
    array = ak._v2.contents.listarray.ListArray(starts, stops, content)

    assert ak.to_list(array) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
    ]
    assert ak.to_list(array.rpad(1, 0, clip=True)) == [[0.0, 1.1, 2.2]]
    assert (
        array.typetracer.rpad(1, 0, clip=True).form == array.rpad(1, 0, clip=True).form
    )
    assert str("option[") + str(array.form.type) + str("]") == str(
        array.rpad(1, 0, clip=True).form.type
    )

    assert ak.to_list(array.rpad(2, 0, clip=True)) == [[0.0, 1.1, 2.2], []]
    assert (
        array.typetracer.rpad(2, 0, clip=True).form == array.rpad(2, 0, clip=True).form
    )
    assert str("option[") + str(array.form.type) + str("]") == str(
        array.rpad(2, 0, clip=True).form.type
    )

    assert ak.to_list(array.rpad(7, 0, clip=True)) == [
        [0.0, 1.1, 2.2],
        [],
        [4.4, 5.5],
        [5.5, 6.6, 7.7],
        [8.8],
        None,
        None,
    ]
    assert (
        array.typetracer.rpad(7, 0, clip=True).form == array.rpad(7, 0, clip=True).form
    )
    assert str("option[") + str(array.form.type) + str("]") == str(
        array.rpad(7, 0, clip=True).form.type
    )

    assert ak.to_list(array.rpad(1, 1, clip=True)) == [
        [0.0],
        [None],
        [4.4],
        [5.5],
        [8.8],
    ]
    assert (
        array.typetracer.rpad(1, 1, clip=True).form == array.rpad(1, 1, clip=True).form
    )

    assert ak.to_list(array.rpad(2, 1, clip=True)) == [
        [0.0, 1.1],
        [None, None],
        [4.4, 5.5],
        [5.5, 6.6],
        [8.8, None],
    ]
    assert (
        array.typetracer.rpad(2, 1, clip=True).form == array.rpad(2, 1, clip=True).form
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
    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]

    assert ak.to_list(backward.rpad(4, 1)) == ak.to_list(indexedarray.rpad(4, 1))
    assert ak.to_list(indexedarray.rpad(1, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert ak.to_list(indexedarray.rpad(2, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None],
        [3.3, 4.4],
        [None, None],
        [0.0, 1.1, 2.2],
    ]
    assert indexedarray.typetracer.rpad(2, 1).form == indexedarray.rpad(2, 1).form
    assert ak.to_list(indexedarray.rpad(3, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None, None],
        [3.3, 4.4, None],
        [None, None, None],
        [0.0, 1.1, 2.2],
    ]
    assert indexedarray.typetracer.rpad(3, 1).form == indexedarray.rpad(3, 1).form
    assert ak.to_list(indexedarray.rpad(4, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert indexedarray.typetracer.rpad(4, 0).form == indexedarray.rpad(4, 0).form
    assert ak.to_list(indexedarray.rpad(5, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert indexedarray.typetracer.rpad(5, 0).form == indexedarray.rpad(5, 0).form
    assert ak.to_list(indexedarray.rpad(6, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
    ]
    assert indexedarray.typetracer.rpad(6, 0).form == indexedarray.rpad(6, 0).form
    assert ak.to_list(indexedarray.rpad(7, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
        None,
    ]
    assert indexedarray.typetracer.rpad(7, 0).form == indexedarray.rpad(7, 0).form


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
    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]

    assert ak.to_list(backward.rpad(4, 1, clip=True)) == ak.to_list(
        indexedarray.rpad(4, 1, clip=True)
    )
    assert ak.to_list(indexedarray.rpad(1, 0, clip=True)) == [[6.6, 7.7, 8.8, 9.9]]
    assert (
        indexedarray.typetracer.rpad(1, 0, clip=True).form
        == indexedarray.rpad(1, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(2, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
    ]
    assert (
        indexedarray.typetracer.rpad(2, 0, clip=True).form
        == indexedarray.rpad(2, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(3, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
    ]
    assert (
        indexedarray.typetracer.rpad(3, 0, clip=True).form
        == indexedarray.rpad(3, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(4, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
    ]
    assert (
        indexedarray.typetracer.rpad(4, 0, clip=True).form
        == indexedarray.rpad(4, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(5, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.rpad(5, 0, clip=True).form
        == indexedarray.rpad(5, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(6, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
    ]
    assert (
        indexedarray.typetracer.rpad(6, 0, clip=True).form
        == indexedarray.rpad(6, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(7, 0, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        None,
        None,
    ]
    assert (
        indexedarray.typetracer.rpad(7, 0, clip=True).form
        == indexedarray.rpad(7, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(8, 0, clip=True)) == [
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
        indexedarray.typetracer.rpad(8, 0, clip=True).form
        == indexedarray.rpad(8, 0, clip=True).form
    )

    assert ak.to_list(indexedarray.rpad(1, 1, clip=True)) == [
        [6.6],
        [5.5],
        [3.3],
        [None],
        [0.0],
    ]
    assert (
        indexedarray.typetracer.rpad(1, 1, clip=True).form
        == indexedarray.rpad(1, 1, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(2, 1, clip=True)) == [
        [6.6, 7.7],
        [5.5, None],
        [3.3, 4.4],
        [None, None],
        [0.0, 1.1],
    ]
    assert (
        indexedarray.typetracer.rpad(2, 1, clip=True).form
        == indexedarray.rpad(2, 1, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(3, 1, clip=True)) == [
        [6.6, 7.7, 8.8],
        [5.5, None, None],
        [3.3, 4.4, None],
        [None, None, None],
        [0.0, 1.1, 2.2],
    ]
    assert (
        indexedarray.typetracer.rpad(3, 1, clip=True).form
        == indexedarray.rpad(3, 1, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(4, 1, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None, None, None],
        [3.3, 4.4, None, None],
        [None, None, None, None],
        [0.0, 1.1, 2.2, None],
    ]
    assert (
        indexedarray.typetracer.rpad(4, 1, clip=True).form
        == indexedarray.rpad(4, 1, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(5, 1, clip=True)) == [
        [6.6, 7.7, 8.8, 9.9, None],
        [5.5, None, None, None, None],
        [3.3, 4.4, None, None, None],
        [None, None, None, None, None],
        [0.0, 1.1, 2.2, None, None],
    ]
    assert (
        indexedarray.typetracer.rpad(5, 1, clip=True).form
        == indexedarray.rpad(5, 1, clip=True).form
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
    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]

    assert ak.to_list(backward.rpad(4, 1)) == ak.to_list(indexedarray.rpad(4, 1))
    assert ak.to_list(indexedarray.rpad(1, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert indexedarray.typetracer.rpad(1, 0).form == indexedarray.rpad(1, 0).form
    assert ak.to_list(indexedarray.rpad(1, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert indexedarray.typetracer.rpad(1, 1).form == indexedarray.rpad(1, 1).form
    assert ak.to_list(indexedarray.rpad(3, 1)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5, None, None],
        [3.3, 4.4, None],
        None,
        [0.0, None, None],
    ]
    assert indexedarray.typetracer.rpad(3, 1).form == indexedarray.rpad(3, 1).form
    assert ak.to_list(indexedarray.rpad(4, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert indexedarray.typetracer.rpad(4, 0).form == indexedarray.rpad(4, 0).form
    assert ak.to_list(indexedarray.rpad(5, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
    ]
    assert indexedarray.typetracer.rpad(5, 0).form == indexedarray.rpad(5, 0).form
    assert ak.to_list(indexedarray.rpad(6, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
        None,
    ]
    assert indexedarray.typetracer.rpad(6, 0).form == indexedarray.rpad(6, 0).form
    assert ak.to_list(indexedarray.rpad(7, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
        None,
        None,
    ]
    assert indexedarray.typetracer.rpad(7, 0).form == indexedarray.rpad(7, 0).form
    assert ak.to_list(indexedarray.rpad(8, 0)) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        None,
        [0.0, None, None],
        None,
        None,
        None,
    ]
    assert indexedarray.typetracer.rpad(8, 0).form == indexedarray.rpad(8, 0).form

    assert ak.to_list(indexedarray.rpad(1, 0, clip=True)) == [[6.6, 7.7, 8.8, 9.9]]
    assert (
        indexedarray.typetracer.rpad(1, 0, clip=True).form
        == indexedarray.rpad(1, 0, clip=True).form
    )
    assert ak.to_list(indexedarray.rpad(1, 1, clip=True)) == [
        [6.6],
        [5.5],
        [3.3],
        None,
        [0.0],
    ]
    assert (
        indexedarray.typetracer.rpad(1, 1, clip=True).form
        == indexedarray.rpad(1, 1, clip=True).form
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

    assert ak.to_list(array.rpad(5, 0)) == [
        {"x": [], "y": [2, 2]},
        {"x": [1.1], "y": [1]},
        {"x": [2.2, 2.2], "y": []},
        None,
        None,
    ]
    assert array.typetracer.rpad(5, 0).form == array.rpad(5, 0).form

    assert ak.to_list(array.rpad(2, 1)) == [
        {"x": [None, None], "y": [2, 2]},
        {"x": [1.1, None], "y": [1, None]},
        {"x": [2.2, 2.2], "y": [None, None]},
    ]
    assert array.typetracer.rpad(2, 1).form == array.rpad(2, 1).form


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
    assert ak.to_list(array) == [[], [2, 2], [1.1], [1], [2.2, 2.2], []]

    assert ak.to_list(array.rpad(7, 0)) == [
        [],
        [2, 2],
        [1.1],
        [1],
        [2.2, 2.2],
        [],
        None,
    ]
    assert array.typetracer.rpad(7, 0).form == array.rpad(7, 0).form

    assert ak.to_list(array.rpad(2, 1)) == [
        [None, None],
        [2, 2],
        [1.1, None],
        [1, None],
        [2.2, 2.2],
        [None, None],
    ]
    assert array.typetracer.rpad(2, 1).form == array.rpad(2, 1).form
