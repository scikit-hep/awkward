# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_array_slice():
    array = ak.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    )
    array = v1_to_v2(array.layout)
    assert ak.to_list(array[[5, 2, 2, 3, 9, 0, 1]]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.typetracer[[5, 2, 2, 3, 9, 0, 1]].form
        == array[[5, 2, 2, 3, 9, 0, 1]].form
    )
    assert ak.to_list(array[np.array([5, 2, 2, 3, 9, 0, 1])]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.typetracer[np.array([5, 2, 2, 3, 9, 0, 1])].form
        == array[np.array([5, 2, 2, 3, 9, 0, 1])].form
    )

    array2 = ak.layout.NumpyArray(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32))
    array2 = v1_to_v2(array2)

    assert ak.to_list(array[array2]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert array.typetracer[array2].form == array[array2].form
    assert ak.to_list(
        array[
            ak.Array(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32), check_valid=True)
        ]
    ) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert (
        array.typetracer[
            ak.Array(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32), check_valid=True)
        ].form
        == array[
            ak.Array(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32), check_valid=True)
        ].form
    )
    assert ak.to_list(array[ak.Array([5, 2, 2, 3, 9, 0, 1], check_valid=True)]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.typetracer[ak.Array([5, 2, 2, 3, 9, 0, 1])].form
        == array[ak.Array([5, 2, 2, 3, 9, 0, 1])].form
    )

    array3 = ak.layout.NumpyArray(
        np.array([False, False, False, False, False, True, False, True, False, True])
    )
    array3 = v1_to_v2(array3)
    assert ak.to_list(array[array3]) == [5.5, 7.7, 9.9]
    assert array.typetracer[array3].form == array[array3].form

    content = ak.layout.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.layout.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)

    indexedarray = v1_to_v2(indexedarray)

    assert ak.to_list(array[indexedarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert array.typetracer[indexedarray].form == array[indexedarray].form
    assert ak.to_list(array[ak.Array(indexedarray, check_valid=True)]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.typetracer[ak.Array(indexedarray)].form
        == array[ak.Array(indexedarray)].form
    )

    emptyarray = ak.layout.EmptyArray()
    emptyarray = v1_to_v2(emptyarray)

    assert ak.to_list(array[emptyarray]) == []
    assert array.typetracer[emptyarray].form == array[emptyarray].form

    content0 = ak.layout.NumpyArray(np.array([5, 2, 2]))
    content1 = ak.layout.NumpyArray(np.array([3, 9, 0, 1]))
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index2 = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(tags, index2, [content0, content1])

    unionarray = v1_to_v2(unionarray)
    assert ak.to_list(array[ak.Array(unionarray, check_valid=True)]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.typetracer[ak.Array(unionarray)].form == array[ak.Array(unionarray)].form
    )

    array = ak.Array(
        np.array([[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]),
        check_valid=True,
    )

    array = v1_to_v2(array.layout)

    numpyarray1 = ak.layout.NumpyArray(np.array([[0, 1], [1, 0]]))
    numpyarray2 = ak.layout.NumpyArray(np.array([[2, 4], [3, 3]]))

    numpyarray1 = v1_to_v2(numpyarray1)
    numpyarray2 = v1_to_v2(numpyarray2)
    assert (
        ak.to_list(
            array[
                numpyarray1,
                numpyarray2,
            ]
        )
        == [[2.2, 9.9], [8.8, 3.3]]
    )
    assert (
        array.typetracer[
            numpyarray1,
            numpyarray2,
        ].form
        == array[
            numpyarray1,
            numpyarray2,
        ].form
    )
    assert ak.to_list(array[numpyarray1]) == [
        [[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]],
        [[5.5, 6.6, 7.7, 8.8, 9.9], [0.0, 1.1, 2.2, 3.3, 4.4]],
    ]
    assert array.typetracer[numpyarray1].form == array[numpyarray1].form


def test_array_slice_1():
    array = ak.Array(
        [
            {"x": 1, "y": 1.1, "z": [1]},
            {"x": 2, "y": 2.2, "z": [2, 2]},
            {"x": 3, "y": 3.3, "z": [3, 3, 3]},
            {"x": 4, "y": 4.4, "z": [4, 4, 4, 4]},
            {"x": 5, "y": 5.5, "z": [5, 5, 5, 5, 5]},
        ],
        check_valid=True,
    ).layout
    array = v1_to_v2(array)
    assert ak.to_list(array[ak.from_iter(["y", "x"], highlevel=False)]) == [
        {"y": 1.1, "x": 1},
        {"y": 2.2, "x": 2},
        {"y": 3.3, "x": 3},
        {"y": 4.4, "x": 4},
        {"y": 5.5, "x": 5},
    ]


@pytest.mark.skip(reason="FIXME: UnionArray as a slice has not been implemented")
def test_array_slice_2():
    array = ak.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    )
    array = v1_to_v2(array.layout)

    content0 = ak.layout.NumpyArray(np.array([5, 2, 2]))
    content1 = ak.layout.NumpyArray(np.array([3, 9, 0, 1]))
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index2 = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(tags, index2, [content0, content1])

    unionarray = v1_to_v2(unionarray)
    assert ak.to_list(array[unionarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert array.typetracer[unionarray].form == array[unionarray].form


def test_new_slices():
    content = ak.layout.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.layout.Index64(np.array([6, 5, -1, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)

    indexedarray = v1_to_v2(indexedarray)
    assert ak.to_list(indexedarray) == [5, 2, None, 3, 9, None, 1]

    assert (
        ak._ext._slice_tostring(indexedarray)
        == "[missing([0, 1, -1, 2, 3, -1, 4], array([5, 2, 3, 9, 1]))]"
    )

    offsets = ak.layout.Index64(np.array([0, 4, 4, 7], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)

    listoffsetarray = v1_to_v2(listoffsetarray)
    assert ak.to_list(listoffsetarray) == [[1, 0, 9, 3], [], [2, 2, 5]]

    assert (
        ak._ext._slice_tostring(listoffsetarray)
        == "[jagged([0, 4, 4, 7], array([1, 0, 9, 3, 2, 2, 5]))]"
    )

    offsets = ak.layout.Index64(np.array([1, 4, 4, 6], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)

    listoffsetarray = v1_to_v2(listoffsetarray)
    assert ak.to_list(listoffsetarray) == [[0, 9, 3], [], [2, 2]]

    assert (
        ak._ext._slice_tostring(listoffsetarray)
        == "[jagged([0, 3, 3, 5], array([0, 9, 3, 2, 2]))]"
    )

    starts = ak.layout.Index64(np.array([1, 99, 5], dtype=np.int64))
    stops = ak.layout.Index64(np.array([4, 99, 7], dtype=np.int64))
    listarray = ak.layout.ListArray64(starts, stops, content)

    listarray = v1_to_v2(listarray)
    assert ak.to_list(listarray) == [[0, 9, 3], [], [2, 5]]

    assert (
        ak._ext._slice_tostring(listarray)
        == "[jagged([0, 3, 3, 5], array([0, 9, 3, 2, 5]))]"
    )


def test_missing():
    array = ak.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    )
    array = v1_to_v2(array.layout)
    array2 = ak.Array([3, 6, None, None, -2, 6], check_valid=True)
    array2 = v1_to_v2(array2.layout)
    assert ak.to_list(array[array2]) == [
        3.3,
        6.6,
        None,
        None,
        8.8,
        6.6,
    ]
    assert array.typetracer[array2].form == array[array2].form

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    regulararray = ak.layout.RegularArray(content, 4, zeros_length=0)

    regulararray = v1_to_v2(regulararray)
    assert ak.to_list(regulararray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]
    array3 = ak.Array([2, 1, 1, None, -1], check_valid=True)
    array3 = v1_to_v2(array3.layout)
    assert ak.to_list(regulararray[array3]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert regulararray.typetracer[array3].form == regulararray[array3].form
    assert ak.to_list(regulararray[:, array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert regulararray.typetracer[:, array3].form == regulararray[:, array3].form
    assert ak.to_list(regulararray[1:, array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert regulararray.typetracer[1:, array3].form == regulararray[1:, array3].form

    assert ak.to_list(
        regulararray[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ]
    ) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert (
        regulararray.typetracer[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
        == regulararray[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
    )
    assert ak.to_list(
        regulararray[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ]
    ) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        regulararray.typetracer[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == regulararray[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )

    assert (
        ak.to_list(
            regulararray[
                1:,
                np.ma.MaskedArray(
                    [2, 1, 1, 999, -1], [False, False, False, True, False]
                ),
            ]
        )
        == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    )
    assert (
        regulararray.typetracer[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == regulararray[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )

    content = ak.layout.NumpyArray(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]])
    )
    content = v1_to_v2(content)
    assert ak.to_list(content[array3]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert content.typetracer[array3].form == content[array3].form
    assert ak.to_list(content[:, array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert content.typetracer[:, array3].form == content[:, array3].form
    assert ak.to_list(content[1:, array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert content.typetracer[1:, array3].form == content[1:, array3].form

    assert ak.to_list(
        content[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ]
    ) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert (
        content.typetracer[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
        == content[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
    )
    assert ak.to_list(
        content[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ]
    ) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        content.typetracer[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == content[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )
    assert (
        ak.to_list(
            content[
                1:,
                np.ma.MaskedArray(
                    [2, 1, 1, 999, -1], [False, False, False, True, False]
                ),
            ]
        )
        == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    )
    assert (
        content.typetracer[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == content[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    offsets = ak.layout.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)

    listoffsetarray = v1_to_v2(listoffsetarray)
    assert ak.to_list(listoffsetarray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert ak.to_list(listoffsetarray[:, array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert listoffsetarray.typetracer[:, array3].form == listoffsetarray[:, array3].form
    assert ak.to_list(listoffsetarray[1:, array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        listoffsetarray.typetracer[1:, array3].form == listoffsetarray[1:, array3].form
    )

    assert ak.to_list(
        listoffsetarray[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ]
    ) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        listoffsetarray.typetracer[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == listoffsetarray[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )
    assert (
        ak.to_list(
            listoffsetarray[
                1:,
                np.ma.MaskedArray(
                    [2, 1, 1, 999, -1], [False, False, False, True, False]
                ),
            ]
        )
        == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    )
    assert (
        listoffsetarray.typetracer[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == listoffsetarray[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )


def test_bool_missing():
    data = [1.1, 2.2, 3.3, 4.4, 5.5]
    array = ak.layout.NumpyArray(np.array(data))

    array = v1_to_v2(array)

    assert (
        ak._ext._slice_tostring(
            ak.Array([True, False, None, True, False], check_valid=True)
        )
        == "[missing([0, -1, 1], array([0, 3]))]"
    )
    assert (
        ak._ext._slice_tostring(ak.Array([None, None, None], check_valid=True))
        == "[missing([-1, -1, -1], array([]))]"
    )

    x1, x2, x3, x4, x5 = True, True, True, False, None
    mask = [x1, x2, x3, x4, x5]
    expected = [m if m is None else x for x, m in zip(data, mask) if m is not False]
    array2 = ak.Array(mask, check_valid=True)
    array2 = v1_to_v2(array2.layout)

    for x1 in [True, False, None]:
        for x2 in [True, False, None]:
            for x3 in [True, False, None]:
                for x4 in [True, False, None]:
                    for x5 in [True, False, None]:
                        mask = [x1, x2, x3, x4, x5]
                        expected = [
                            m if m is None else x
                            for x, m in zip(data, mask)
                            if m is not False
                        ]
                        array2 = ak.Array(mask, check_valid=True)
                        array2 = v1_to_v2(array2.layout)
                        assert ak.to_list(array[array2]) == expected
                        assert array.typetracer[array2].form == array[array2].form


def test_bool_missing2():
    array = ak.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    )
    array2 = ak.Array([3, 6, None, None, -2, 6], check_valid=True)

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array[array2]) == [
        3.3,
        6.6,
        None,
        None,
        8.8,
        6.6,
    ]
    assert array.typetracer[array2].form == array[array2].form

    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    array1 = v1_to_v2(array)

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    regulararray = ak.layout.RegularArray(content, 4, zeros_length=0)

    assert ak.to_list(regulararray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]

    regulararray = v1_to_v2(regulararray)

    array1 = v1_to_v2(ak.from_iter([True, None, False, True], highlevel=False))

    assert ak.to_list(regulararray[:, array1]) == [
        [0.0, None, 3.3],
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert regulararray.typetracer[:, array1].form == regulararray[:, array1].form

    assert ak.to_list(regulararray[1:, array1]) == [
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert regulararray.typetracer[1:, array1].form == regulararray[1:, array1].form

    content = ak.layout.NumpyArray(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]])
    )
    content = v1_to_v2(content)

    assert ak.to_list(content[:, array1]) == [
        [0.0, None, 3.3],
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert content.typetracer[:, array1].form == content[:, array1].form

    assert ak.to_list(content[1:, array1]) == [[4.4, None, 7.7], [8.8, None, 11.1]]
    assert content.typetracer[1:, array1].form == content[1:, array1].form

    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    offsets = ak.layout.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)

    listoffsetarray = v1_to_v2(listoffsetarray)

    assert ak.to_list(listoffsetarray[:, array1]) == [
        [0.0, None, 3.3],
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert listoffsetarray.typetracer[:, array1].form == listoffsetarray[:, array1].form

    assert ak.to_list(listoffsetarray[1:, array1]) == [
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert (
        listoffsetarray.typetracer[1:, array1].form == listoffsetarray[1:, array1].form
    )


def test_records_missing():
    array = ak.Array(
        [
            {"x": 0, "y": 0.0},
            {"x": 1, "y": 1.1},
            {"x": 2, "y": 2.2},
            {"x": 3, "y": 3.3},
            {"x": 4, "y": 4.4},
            {"x": 5, "y": 5.5},
            {"x": 6, "y": 6.6},
            {"x": 7, "y": 7.7},
            {"x": 8, "y": 8.8},
            {"x": 9, "y": 9.9},
        ],
        check_valid=True,
    )
    array2 = ak.Array([3, 1, None, 1, 7], check_valid=True)

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array[array2]) == [
        {"x": 3, "y": 3.3},
        {"x": 1, "y": 1.1},
        None,
        {"x": 1, "y": 1.1},
        {"x": 7, "y": 7.7},
    ]
    assert array.typetracer[array2].form == array[array2].form

    array = ak.Array(
        [
            [
                {"x": 0, "y": 0.0},
                {"x": 1, "y": 1.1},
                {"x": 2, "y": 2.2},
                {"x": 3, "y": 3.3},
            ],
            [
                {"x": 4, "y": 4.4},
                {"x": 5, "y": 5.5},
                {"x": 6, "y": 6.6},
                {"x": 7, "y": 7.7},
                {"x": 8, "y": 8.8},
                {"x": 9, "y": 9.9},
            ],
        ],
        check_valid=True,
    )
    array2 = ak.Array([1, None, 2, -1], check_valid=True)

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array[:, array2]) == [
        [{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [{"x": 5, "y": 5.5}, None, {"x": 6, "y": 6.6}, {"x": 9, "y": 9.9}],
    ]
    assert array.typetracer[:, array2].form == array[:, array2].form

    array = ak.Array(
        [
            {"x": [0, 1, 2, 3], "y": [0.0, 1.1, 2.2, 3.3]},
            {"x": [4, 5, 6, 7], "y": [4.4, 5.5, 6.6, 7.7]},
            {"x": [8, 9, 10, 11], "y": [8.8, 9.9, 10.0, 11.1]},
        ],
        check_valid=True,
    )

    array = v1_to_v2(array.layout)

    assert ak.to_list(array[:, array2]) == [
        {"x": [1, None, 2, 3], "y": [1.1, None, 2.2, 3.3]},
        {"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]},
        {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]},
    ]
    assert array.typetracer[:, array2].form == array[:, array2].form
    assert ak.to_list(array[1:, array2]) == [
        {"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]},
        {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]},
    ]
    assert array.typetracer[1:, array2].form == array[1:, array2].form


def test_jagged():
    array = ak.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    )
    array2 = ak.Array([[0, -1], [], [-1, 0], [-1], [1, 1, -2, 0]], check_valid=True)

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array[array2]) == [
        [1.1, 3.3],
        [],
        [5.5, 4.4],
        [6.6],
        [8.8, 8.8, 8.8, 7.7],
    ]
    assert array.typetracer[array2].form == array[array2].form


def test_double_jagged():
    array = ak.Array(
        [[[0, 1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10, 11, 12, 13]]], check_valid=True
    )
    array2 = ak.Array(
        [[[2, 1, 0], [-1]], [[-1, -2, -3], [2, 1, 1, 3]]], check_valid=True
    )

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array[array2]) == [
        [[2, 1, 0], [5]],
        [[8, 7, 6], [11, 10, 10, 12]],
    ]
    assert array.typetracer[array2].form == array[array2].form

    content = ak.from_iter(
        [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12, 13]], highlevel=False
    )
    regulararray = ak.layout.RegularArray(content, 2, zeros_length=0)
    regulararray = v1_to_v2(regulararray)

    array1 = ak.Array([[2, 1, 0], [-1]], check_valid=True)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(regulararray[:, array1]) == [[[2, 1, 0], [5]], [[8, 7, 6], [13]]]
    assert regulararray.typetracer[:, array1].form == regulararray[:, array1].form
    assert ak.to_list(regulararray[1:, array1]) == [[[8, 7, 6], [13]]]
    assert regulararray.typetracer[1:, array1].form == regulararray[1:, array1].form

    offsets = ak.layout.Index64(np.array([0, 2, 4], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)
    listoffsetarray = v1_to_v2(listoffsetarray)
    assert ak.to_list(listoffsetarray[:, array1]) == [
        [[2, 1, 0], [5]],
        [[8, 7, 6], [13]],
    ]
    assert listoffsetarray.typetracer[:, array1].form == listoffsetarray[:, array1].form
    assert ak.to_list(listoffsetarray[1:, array1]) == [[[8, 7, 6], [13]]]
    assert (
        listoffsetarray.typetracer[1:, array1].form == listoffsetarray[1:, array1].form
    )


def test_masked_jagged():
    array = ak.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    )
    array1 = ak.Array([[-1, -2], None, [], None, [-2, 0]], check_valid=True)

    array = v1_to_v2(array.layout)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(array[array1]) == [[3.3, 2.2], None, [], None, [8.8, 7.7]]
    assert array.typetracer[array1].form == array[array1].form


def test_jagged_masked():
    array = ak.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    )
    array1 = ak.Array([[-1, None], [], [None, 0], [None], [1]], check_valid=True)

    array = v1_to_v2(array.layout)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(array[array1]) == [[3.3, None], [], [None, 4.4], [None], [8.8]]
    assert array.typetracer[array1].form == array[array1].form


def test_regular_regular():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5))
    regulararray1 = ak.layout.RegularArray(content, 5, zeros_length=0)
    regulararray2 = ak.layout.RegularArray(regulararray1, 3, zeros_length=0)

    regulararray2 = v1_to_v2(regulararray2)
    array1 = ak.Array(
        [[[2], [1, -2], [-1, 2, 0]], [[-3], [-4, 3], [-5, -3, 4]]],
        check_valid=True,
    )
    array2 = ak.Array(
        [[[2], [1, -2], [-1, None, 0]], [[-3], [-4, 3], [-5, None, 4]]],
        check_valid=True,
    )
    array1 = v1_to_v2(array1.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(regulararray2[array1]) == [
        [[2], [6, 8], [14, 12, 10]],
        [[17], [21, 23], [25, 27, 29]],
    ]
    assert regulararray2.typetracer[array1].form == regulararray2[array1].form

    assert ak.to_list(regulararray2[array2]) == [
        [[2], [6, 8], [14, None, 10]],
        [[17], [21, 23], [25, None, 29]],
    ]
    assert regulararray2.typetracer[array2].form == regulararray2[array2].form


def test_masked_of_jagged_of_whatever():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5))
    regulararray1 = ak.layout.RegularArray(content, 5, zeros_length=0)
    regulararray2 = ak.layout.RegularArray(regulararray1, 3, zeros_length=0)

    regulararray2 = v1_to_v2(regulararray2)
    array1 = ak.Array(
        [[[2], None, [-1, 2, 0]], [[-3], None, [-5, -3, 4]]], check_valid=True
    )
    array2 = ak.Array(
        [[[2], None, [-1, None, 0]], [[-3], None, [-5, None, 4]]],
        check_valid=True,
    )
    array1 = v1_to_v2(array1.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(regulararray2[array1]) == [
        [[2], None, [14, 12, 10]],
        [[17], None, [25, 27, 29]],
    ]
    assert regulararray2.typetracer[array1].form == regulararray2[array1].form

    assert ak.to_list(regulararray2[array2]) == [
        [[2], None, [14, None, 10]],
        [[17], None, [25, None, 29]],
    ]
    assert regulararray2.typetracer[array2].form == regulararray2[array2].form


def test_emptyarray():
    content = ak.layout.EmptyArray()
    offsets = ak.layout.Index64(np.array([0, 0, 0, 0, 0], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, content)

    listoffsetarray = v1_to_v2(listoffsetarray)
    array1 = ak.Array([[], [], [], []], check_valid=True)
    array2 = ak.Array([[], [None], [], []], check_valid=True)
    array3 = ak.Array([[], [], None, []], check_valid=True)
    array4 = ak.Array([[], [None], None, []], check_valid=True)
    array5 = ak.Array([[], [0], [], []], check_valid=True)

    array1 = v1_to_v2(array1.layout)
    array2 = v1_to_v2(array2.layout)
    array3 = v1_to_v2(array3.layout)
    array4 = v1_to_v2(array4.layout)
    array5 = v1_to_v2(array5.layout)
    assert ak.to_list(listoffsetarray) == [[], [], [], []]

    assert ak.to_list(listoffsetarray[array1]) == [[], [], [], []]
    assert listoffsetarray.typetracer[array1].form == listoffsetarray[array1].form

    assert ak.to_list(listoffsetarray[array2]) == [[], [None], [], []]
    assert listoffsetarray.typetracer[array2].form == listoffsetarray[array2].form
    assert ak.to_list(listoffsetarray[array3]) == [[], [], None, []]
    assert listoffsetarray.typetracer[array3].form == listoffsetarray[array3].form
    assert ak.to_list(listoffsetarray[array4]) == [[], [None], None, []]
    assert listoffsetarray.typetracer[array4].form == listoffsetarray[array4].form

    with pytest.raises(ValueError):
        listoffsetarray[array5]


def test_numpyarray():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    array2 = ak.Array([[[], [], []], [], [[], []]], check_valid=True)

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)
    with pytest.raises(IndexError):
        array[array2]


def test_record():
    array = ak.Array(
        [
            {"x": [0, 1, 2], "y": [0.0, 1.1, 2.2, 3.3]},
            {"x": [3, 4, 5, 6], "y": [4.4, 5.5]},
            {"x": [7, 8], "y": [6.6, 7.7, 8.8, 9.9]},
        ],
        check_valid=True,
    )
    array2 = ak.Array([[-1, 1], [0, 0, 1], [-1, -2]], check_valid=True)
    array3 = ak.Array([[-1, 1], [0, 0, None, 1], [-1, -2]], check_valid=True)
    array4 = ak.Array([[-1, 1], None, [-1, -2]], check_valid=True)

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)
    array3 = v1_to_v2(array3.layout)
    array4 = v1_to_v2(array4.layout)

    assert ak.to_list(array[array2]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        {"x": [3, 3, 4], "y": [4.4, 4.4, 5.5]},
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert array.typetracer[array2].form == array[array2].form
    assert ak.to_list(array[array3]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        {"x": [3, 3, None, 4], "y": [4.4, 4.4, None, 5.5]},
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert array.typetracer[array3].form == array[array3].form
    assert ak.to_list(array[array4]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        None,
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert array.typetracer[array4].form == array[array4].form


def test_indexedarray():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.layout.Index64(np.array([3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, array)

    indexedarray = v1_to_v2(indexedarray)

    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
    ]

    array1 = ak.Array([[0, -1], [0], [], [1, 1]], check_valid=True)
    array1 = v1_to_v2(array1.layout)
    assert ak.to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], [], [1.1, 1.1]]
    assert indexedarray.typetracer[array1].form == indexedarray[array1].form

    array1 = ak.Array([[0, -1], [0], [None], [1, None, 1]], check_valid=True)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(indexedarray[array1]) == [
        [6.6, 9.9],
        [5.5],
        [None],
        [1.1, None, 1.1],
    ]
    assert indexedarray.typetracer[array1].form == indexedarray[array1].form

    array1 = ak.Array([[0, -1], [0], None, [1, 1]], check_valid=True)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], None, [1.1, 1.1]]
    assert indexedarray.typetracer[array1].form == indexedarray[array1].form

    array1 = ak.Array([[0, -1], [0], None, [None]], check_valid=True)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], None, [None]]
    assert indexedarray.typetracer[array1].form == indexedarray[array1].form

    index = ak.layout.Index64(np.array([3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, array)

    indexedarray = v1_to_v2(indexedarray)

    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
    ]

    assert ak.to_list(
        indexedarray[ak.Array([[0, -1], [0], [], [1, 1]], check_valid=True)]
    ) == [[6.6, 9.9], [5.5], [], [1.1, 1.1]]
    assert (
        indexedarray.typetracer[ak.Array([[0, -1], [0], [], [1, 1]])].form
        == indexedarray[ak.Array([[0, -1], [0], [], [1, 1]])].form
    )

    array1 = ak.Array([[0, -1], [0], [None], [1, None, 1]], check_valid=True)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(indexedarray[array1]) == [
        [6.6, 9.9],
        [5.5],
        [None],
        [1.1, None, 1.1],
    ]
    assert indexedarray.typetracer[array1].form == indexedarray[array1].form

    array1 = ak.Array([[0, -1], [0], None, []], check_valid=True)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], None, []]
    assert indexedarray.typetracer[array1].form == indexedarray[array1].form

    array1 = ak.Array([[0, -1], [0], None, [1, None, 1]], check_valid=True)
    array1 = v1_to_v2(array1.layout)

    assert ak.to_list(indexedarray[array1]) == [
        [6.6, 9.9],
        [5.5],
        None,
        [1.1, None, 1.1],
    ]
    assert indexedarray.typetracer[array1].form == indexedarray[array1].form


def test_indexedarray2():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.layout.Index64(np.array([3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, array)
    array = ak.Array([[0, -1], [0], None, [1, 1]])

    indexedarray = v1_to_v2(indexedarray)
    array = v1_to_v2(array.layout)

    assert ak.to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert ak.to_list(indexedarray[array]) == [
        [6.6, 9.9],
        [5.5],
        None,
        [1.1, 1.1],
    ]
    assert indexedarray.typetracer[array].form == indexedarray[array].form


def test_indexedarray2b():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.layout.Index64(np.array([0, -1, 2, 3], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, array)
    array = ak.Array([[1, 1], None, [0], [0, -1]])

    indexedarray = v1_to_v2(indexedarray)
    array = v1_to_v2(array.layout)

    assert ak.to_list(indexedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(indexedarray[array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert indexedarray.typetracer[array].form == indexedarray[array].form


def test_bytemaskedarray2b():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.Index8(np.array([0, 1, 0, 0], dtype=np.int8))
    maskedarray = ak.layout.ByteMaskedArray(mask, array, valid_when=False)
    array = ak.Array([[1, 1], None, [0], [0, -1]])

    maskedarray = v1_to_v2(maskedarray)
    array = v1_to_v2(array.layout)

    assert ak.to_list(maskedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(maskedarray[array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert maskedarray.typetracer[array].form == maskedarray[array].form


def test_bitmaskedarray2b():
    array = ak.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.layout.IndexU8(np.array([66], dtype=np.uint8))
    maskedarray = ak.layout.BitMaskedArray(
        mask, array, valid_when=False, length=4, lsb_order=True
    )  # lsb_order is irrelevant in this example
    array = ak.Array([[1, 1], None, [0], [0, -1]])
    maskedarray = v1_to_v2(maskedarray)
    array = v1_to_v2(array.layout)

    assert ak.to_list(maskedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(maskedarray[array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert maskedarray.typetracer[array].form == maskedarray[array].form


def test_indexedarray3():
    array = ak.Array([0.0, 1.1, 2.2, None, 4.4, None, None, 7.7])
    assert ak.to_list(array[ak.Array([4, 3, 2])]) == [4.4, None, 2.2]
    assert ak.to_list(array[ak.Array([4, 3, 2, None, 1])]) == [
        4.4,
        None,
        2.2,
        None,
        1.1,
    ]

    array = ak.Array([[0.0, 1.1, None, 2.2], [3.3, None, 4.4], [5.5]])
    array2 = ak.Array([[3, 2, 2, 1], [1, 2], []])

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array[array2]) == [
        [2.2, None, None, 1.1],
        [None, 4.4],
        [],
    ]
    assert array.typetracer[array2].form == array[array2].form

    array = ak.Array([[0.0, 1.1, 2.2], [3.3, 4.4], None, [5.5]])
    array2 = ak.Array([3, 2, 1])
    array3 = ak.Array([3, 2, 1, None, 0])
    array4 = ak.Array([[2, 1, 1, 0], [1], None, [0]])
    array5 = ak.Array([[2, 1, 1, 0], None, [1], [0]])
    array6 = ak.Array([[2, 1, 1, 0], None, [1], [0], None])

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)
    array3 = v1_to_v2(array3.layout)
    array4 = v1_to_v2(array4.layout)
    array5 = v1_to_v2(array5.layout)
    array6 = v1_to_v2(array6.layout)

    assert ak.to_list(array[array2]) == [[5.5], None, [3.3, 4.4]]
    assert array.typetracer[array2].form == array[array2].form
    assert ak.to_list(array[array3]) == [
        [5.5],
        None,
        [3.3, 4.4],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert array.typetracer[array3].form == array[array3].form

    assert (ak.to_list(array[array4])) == [
        [2.2, 1.1, 1.1, 0.0],
        [4.4],
        None,
        [5.5],
    ]
    assert array.typetracer[array4].form == array[array4].form

    assert ak.to_list(array[array5]) == [
        [2.2, 1.1, 1.1, 0],
        None,
        None,
        [5.5],
    ]
    assert array.typetracer[array5].form == array[array5].form
    with pytest.raises(IndexError):
        array[array6]


def test_sequential():
    array = ak.Array(np.arange(2 * 3 * 5).reshape(2, 3, 5).tolist(), check_valid=True)
    array2 = ak.Array([[2, 1, 0], [2, 1, 0]], check_valid=True)

    array = v1_to_v2(array.layout)
    array2 = v1_to_v2(array2.layout)

    assert ak.to_list(array[array2]) == [
        [[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
        [[25, 26, 27, 28, 29], [20, 21, 22, 23, 24], [15, 16, 17, 18, 19]],
    ]
    assert array.typetracer[array2].form == array[array2].form
    assert ak.to_list(array[array2, :2]) == [
        [[10, 11], [5, 6], [0, 1]],
        [[25, 26], [20, 21], [15, 16]],
    ]
    assert array.typetracer[array2, :2].form == array[array2, :2].form


def test_union():
    one = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = ak.from_iter(
        [[6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]], highlevel=False
    )
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(tags, index, [one, two])
    array = ak.Array([[0, -1], [], [1, 1], [], [-1], [], [1, -2, -1]], check_valid=True)

    unionarray = v1_to_v2(unionarray)
    array = v1_to_v2(array.layout)
    assert ak.to_list(unionarray) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8],
        [],
        [9.9, 10.0, 11.1, 12.2],
    ]


@pytest.mark.skip(reason="FIXME: simplify_uniontype needs to be implemented")
def test_union_2():
    one = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = ak.from_iter(
        [[6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]], highlevel=False
    )
    tags = ak.layout.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.layout.UnionArray8_64(tags, index, [one, two])
    array = ak.Array([[0, -1], [], [1, 1], [], [-1], [], [1, -2, -1]], check_valid=True)

    unionarray = v1_to_v2(unionarray)
    array = v1_to_v2(array.layout)
    assert ak.to_list(unionarray[array]) == [
        [1.1, 3.3],
        [],
        [5.5, 5.5],
        [],
        [8.8],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert unionarray.typetracer[array].form == unionarray[array].form


def test_jagged_mask():
    array = ak.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    )
    array = v1_to_v2(array.layout)
    assert ak.to_list(
        array[[[True, True, True], [], [True, True], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.typetracer[
            [[True, True, True], [], [True, True], [True], [True, True, True]]
        ].form
        == array[
            [[True, True, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert ak.to_list(
        array[[[False, True, True], [], [True, True], [True], [True, True, True]]]
    ) == [[2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.typetracer[
            [[False, True, True], [], [True, True], [True], [True, True, True]]
        ].form
        == array[
            [[False, True, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert ak.to_list(
        array[[[True, False, True], [], [True, True], [True], [True, True, True]]]
    ) == [[1.1, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.typetracer[
            [[True, False, True], [], [True, True], [True], [True, True, True]]
        ].form
        == array[
            [[True, False, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert ak.to_list(
        array[[[True, True, True], [], [False, True], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.typetracer[
            [[True, True, True], [], [False, True], [True], [True, True, True]]
        ].form
        == array[
            [[True, True, True], [], [False, True], [True], [True, True, True]]
        ].form
    )
    assert ak.to_list(
        array[[[True, True, True], [], [False, False], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.typetracer[
            [[True, True, True], [], [False, False], [True], [True, True, True]]
        ].form
        == array[
            [[True, True, True], [], [False, False], [True], [True, True, True]]
        ].form
    )


def test_jagged_missing_mask():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    array = v1_to_v2(array.layout)

    assert ak.to_list(array[[[True, True, True], [], [True, True]]]) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.typetracer[[[True, True, True], [], [True, True]]].form
        == array[[[True, True, True], [], [True, True]]].form
    )
    assert ak.to_list(array[[[True, False, True], [], [True, True]]]) == [
        [1.1, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.typetracer[[[True, False, True], [], [True, True]]].form
        == array[[[True, False, True], [], [True, True]]].form
    )
    assert ak.to_list(array[[[True, True, False], [], [False, None]]]) == [
        [1.1, 2.2],
        [],
        [None],
    ]
    assert (
        array.typetracer[[[True, True, False], [], [False, None]]].form
        == array[[[True, True, False], [], [False, None]]].form
    )
    assert ak.to_list(array[[[True, True, False], [], [True, None]]]) == [
        [1.1, 2.2],
        [],
        [4.4, None],
    ]
    assert (
        array.typetracer[[[True, True, False], [], [True, None]]].form
        == array[[[True, True, False], [], [True, None]]].form
    )

    assert ak.to_list(array[[[True, None, True], [], [True, True]]]) == [
        [1.1, None, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.typetracer[[[True, None, True], [], [True, True]]].form
        == array[[[True, None, True], [], [True, True]]].form
    )
    assert ak.to_list(array[[[True, None, False], [], [True, True]]]) == [
        [1.1, None],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.typetracer[[[True, None, False], [], [True, True]]].form
        == array[[[True, None, False], [], [True, True]]].form
    )

    assert ak.to_list(array[[[False, None, False], [], [True, True]]]) == [
        [None],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.typetracer[[[False, None, False], [], [True, True]]].form
        == array[[[False, None, False], [], [True, True]]].form
    )
    assert ak.to_list(array[[[True, True, False], [], [False, True]]]) == [
        [1.1, 2.2],
        [],
        [5.5],
    ]
    assert (
        array.typetracer[[[True, True, False], [], [False, True]]].form
        == array[[[True, True, False], [], [False, True]]].form
    )
    assert ak.to_list(array[[[True, True, None], [], [False, True]]]) == [
        [1.1, 2.2, None],
        [],
        [5.5],
    ]
    assert (
        array.typetracer[[[True, True, None], [], [False, True]]].form
        == array[[[True, True, None], [], [False, True]]].form
    )
    assert ak.to_list(array[[[True, True, False], [None], [False, True]]]) == [
        [1.1, 2.2],
        [None],
        [5.5],
    ]
    assert (
        array.typetracer[[[True, True, False], [None], [False, True]]].form
        == array[[[True, True, False], [None], [False, True]]].form
    )


def test_array_boolean_to_int():
    a = v1_to_v2(
        ak.from_iter(
            [[True, True, True], [], [True, True], [True], [True, True, True, True]],
            highlevel=False,
        )
    )
    b = ak._v2._slicing.prepare_tuple_bool_to_int(a)
    assert ak.to_list(b) == [[0, 1, 2], [], [0, 1], [0], [0, 1, 2, 3]]

    a = v1_to_v2(
        ak.from_iter(
            [
                [True, True, False],
                [],
                [True, False],
                [False],
                [True, True, True, False],
            ],
            highlevel=False,
        )
    )
    b = ak._v2._slicing.prepare_tuple_bool_to_int(a)
    assert ak.to_list(b) == [[0, 1], [], [0], [], [0, 1, 2]]

    a = v1_to_v2(
        ak.from_iter(
            [
                [False, True, True],
                [],
                [False, True],
                [False],
                [False, True, True, True],
            ],
            highlevel=False,
        )
    )
    b = ak._v2._slicing.prepare_tuple_bool_to_int(a)
    assert ak.to_list(b) == [[1, 2], [], [1], [], [1, 2, 3]]

    a = v1_to_v2(
        ak.from_iter(
            [[True, True, None], [], [True, None], [None], [True, True, True, None]],
            highlevel=False,
        )
    )
    b = ak._v2._slicing.prepare_tuple_bool_to_int(a)
    assert ak.to_list(b) == [[0, 1, None], [], [0, None], [None], [0, 1, 2, None]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(6).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = v1_to_v2(
        ak.from_iter(
            [[None, True, True], [], [None, True], [None], [None, True, True, True]],
            highlevel=False,
        )
    )
    b = ak._v2._slicing.prepare_tuple_bool_to_int(a)
    assert ak.to_list(b) == [[None, 1, 2], [], [None, 1], [None], [None, 1, 2, 3]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(6).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = v1_to_v2(
        ak.from_iter(
            [[False, True, None], [], [False, None], [None], [False, True, True, None]],
            highlevel=False,
        )
    )
    b = ak._v2._slicing.prepare_tuple_bool_to_int(a)
    assert ak.to_list(b) == [[1, None], [], [None], [None], [1, 2, None]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(3).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = v1_to_v2(
        ak.from_iter(
            [[None, True, False], [], [None, False], [None], [None, True, True, False]],
            highlevel=False,
        )
    )
    b = ak._v2._slicing.prepare_tuple_bool_to_int(a)
    assert ak.to_list(b) == [[None, 1], [], [None], [None], [None, 1, 2]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(3).tolist()  # kernels expect nonnegative entries to be arange
    )
