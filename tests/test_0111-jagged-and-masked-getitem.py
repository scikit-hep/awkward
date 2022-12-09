# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_array_slice_with_union():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout

    content0 = ak.contents.NumpyArray(np.array([5, 2, 2]))
    content1 = ak.contents.NumpyArray(np.array([3, 9, 0, 1]))
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index2 = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray.simplified(tags, index2, [content0, content1])

    assert to_list(array[ak.highlevel.Array(unionarray, check_valid=True)]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.to_typetracer()[ak.highlevel.Array(unionarray)].form
        == array[ak.highlevel.Array(unionarray)].form
    )


def test_array_slice():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout
    assert to_list(array[[5, 2, 2, 3, 9, 0, 1]]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.to_typetracer()[[5, 2, 2, 3, 9, 0, 1]].form
        == array[[5, 2, 2, 3, 9, 0, 1]].form
    )
    assert to_list(array[np.array([5, 2, 2, 3, 9, 0, 1])]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.to_typetracer()[np.array([5, 2, 2, 3, 9, 0, 1])].form
        == array[np.array([5, 2, 2, 3, 9, 0, 1])].form
    )

    array2 = ak.contents.NumpyArray(np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32))

    assert to_list(array[array2]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert array.to_typetracer()[array2].form == array[array2].form
    assert to_list(
        array[
            ak.highlevel.Array(
                np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32), check_valid=True
            )
        ]
    ) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert (
        array.to_typetracer()[
            ak.highlevel.Array(
                np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32), check_valid=True
            )
        ].form
        == array[
            ak.highlevel.Array(
                np.array([5, 2, 2, 3, 9, 0, 1], dtype=np.int32), check_valid=True
            )
        ].form
    )
    assert to_list(
        array[ak.highlevel.Array([5, 2, 2, 3, 9, 0, 1], check_valid=True)]
    ) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.to_typetracer()[ak.highlevel.Array([5, 2, 2, 3, 9, 0, 1])].form
        == array[ak.highlevel.Array([5, 2, 2, 3, 9, 0, 1])].form
    )

    array3 = ak.contents.NumpyArray(
        np.array([False, False, False, False, False, True, False, True, False, True])
    )
    assert to_list(array[array3]) == [5.5, 7.7, 9.9]
    assert array.to_typetracer()[array3].form == array[array3].form

    content = ak.contents.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.index.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)

    assert to_list(array[indexedarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert array.to_typetracer()[indexedarray].form == array[indexedarray].form
    assert to_list(array[ak.highlevel.Array(indexedarray, check_valid=True)]) == [
        5.5,
        2.2,
        2.2,
        3.3,
        9.9,
        0.0,
        1.1,
    ]
    assert (
        array.to_typetracer()[ak.highlevel.Array(indexedarray)].form
        == array[ak.highlevel.Array(indexedarray)].form
    )

    emptyarray = ak.contents.EmptyArray()

    assert to_list(array[emptyarray]) == []
    assert array.to_typetracer()[emptyarray].form == array[emptyarray].form

    array = ak.highlevel.Array(
        np.array([[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]),
        check_valid=True,
    ).layout

    numpyarray1 = ak.contents.NumpyArray(np.array([[0, 1], [1, 0]]))
    numpyarray2 = ak.contents.NumpyArray(np.array([[2, 4], [3, 3]]))

    assert to_list(
        array[
            numpyarray1,
            numpyarray2,
        ]
    ) == [[2.2, 9.9], [8.8, 3.3]]
    assert (
        array.to_typetracer()[
            numpyarray1,
            numpyarray2,
        ].form
        == array[
            numpyarray1,
            numpyarray2,
        ].form
    )
    assert to_list(array[numpyarray1]) == [
        [[0.0, 1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]],
        [[5.5, 6.6, 7.7, 8.8, 9.9], [0.0, 1.1, 2.2, 3.3, 4.4]],
    ]
    assert array.to_typetracer()[numpyarray1].form == array[numpyarray1].form


def test_array_slice_1():
    array = ak.highlevel.Array(
        [
            {"x": 1, "y": 1.1, "z": [1]},
            {"x": 2, "y": 2.2, "z": [2, 2]},
            {"x": 3, "y": 3.3, "z": [3, 3, 3]},
            {"x": 4, "y": 4.4, "z": [4, 4, 4, 4]},
            {"x": 5, "y": 5.5, "z": [5, 5, 5, 5, 5]},
        ],
        check_valid=True,
    ).layout
    assert to_list(array[ak.operations.from_iter(["y", "x"], highlevel=False)]) == [
        {"y": 1.1, "x": 1},
        {"y": 2.2, "x": 2},
        {"y": 3.3, "x": 3},
        {"y": 4.4, "x": 4},
        {"y": 5.5, "x": 5},
    ]


def test_array_slice_2():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout

    content0 = ak.contents.NumpyArray(np.array([5, 2, 2]))
    content1 = ak.contents.NumpyArray(np.array([3, 9, 0, 1]))
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index2 = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray.simplified(tags, index2, [content0, content1])

    assert to_list(array[unionarray]) == [5.5, 2.2, 2.2, 3.3, 9.9, 0.0, 1.1]
    assert array.to_typetracer()[unionarray].form == array[unionarray].form


def test_new_slices():
    content = ak.contents.NumpyArray(np.array([1, 0, 9, 3, 2, 2, 5], dtype=np.int64))
    index = ak.index.Index64(np.array([6, 5, -1, 3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, content)

    assert to_list(indexedarray) == [5, 2, None, 3, 9, None, 1]

    offsets = ak.index.Index64(np.array([0, 4, 4, 7], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray) == [[1, 0, 9, 3], [], [2, 2, 5]]

    offsets = ak.index.Index64(np.array([1, 4, 4, 6], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray) == [[0, 9, 3], [], [2, 2]]

    starts = ak.index.Index64(np.array([1, 99, 5], dtype=np.int64))
    stops = ak.index.Index64(np.array([4, 99, 7], dtype=np.int64))
    listarray = ak.contents.ListArray(starts, stops, content)

    assert to_list(listarray) == [[0, 9, 3], [], [2, 5]]


def test_missing():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout
    array2 = ak.highlevel.Array([3, 6, None, None, -2, 6], check_valid=True).layout
    assert to_list(array[array2]) == [
        3.3,
        6.6,
        None,
        None,
        8.8,
        6.6,
    ]
    assert array.to_typetracer()[array2].form == array[array2].form

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    regulararray = ak.contents.RegularArray(content, 4, zeros_length=0)

    assert to_list(regulararray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]
    array3 = ak.highlevel.Array([2, 1, 1, None, -1], check_valid=True).layout
    assert to_list(regulararray[array3]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert regulararray.to_typetracer()[array3].form == regulararray[array3].form
    assert to_list(regulararray[:, array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert regulararray.to_typetracer()[:, array3].form == regulararray[:, array3].form
    assert to_list(regulararray[1:, array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        regulararray.to_typetracer()[1:, array3].form == regulararray[1:, array3].form
    )

    assert to_list(
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
        regulararray.to_typetracer()[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
        == regulararray[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
    )
    assert to_list(
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
        regulararray.to_typetracer()[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == regulararray[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )

    assert to_list(
        regulararray[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ]
    ) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert (
        regulararray.to_typetracer()[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == regulararray[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )

    content = ak.contents.NumpyArray(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]])
    )
    assert to_list(content[array3]) == [
        [8.8, 9.9, 10.0, 11.1],
        [4.4, 5.5, 6.6, 7.7],
        [4.4, 5.5, 6.6, 7.7],
        None,
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert content.to_typetracer()[array3].form == content[array3].form
    assert to_list(content[:, array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert content.to_typetracer()[:, array3].form == content[:, array3].form
    assert to_list(content[1:, array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert content.to_typetracer()[1:, array3].form == content[1:, array3].form

    assert to_list(
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
        content.to_typetracer()[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
        == content[
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False])
        ].form
    )
    assert to_list(
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
        content.to_typetracer()[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == content[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )
    assert to_list(
        content[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ]
    ) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert (
        content.to_typetracer()[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == content[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    offsets = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]
    assert to_list(listoffsetarray[:, array3]) == [
        [2.2, 1.1, 1.1, None, 3.3],
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        listoffsetarray.to_typetracer()[:, array3].form
        == listoffsetarray[:, array3].form
    )
    assert to_list(listoffsetarray[1:, array3]) == [
        [6.6, 5.5, 5.5, None, 7.7],
        [10.0, 9.9, 9.9, None, 11.1],
    ]
    assert (
        listoffsetarray.to_typetracer()[1:, array3].form
        == listoffsetarray[1:, array3].form
    )

    assert to_list(
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
        listoffsetarray.to_typetracer()[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
        == listoffsetarray[
            :,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ].form
    )
    assert to_list(
        listoffsetarray[
            1:,
            np.ma.MaskedArray([2, 1, 1, 999, -1], [False, False, False, True, False]),
        ]
    ) == [[6.6, 5.5, 5.5, None, 7.7], [10.0, 9.9, 9.9, None, 11.1]]
    assert (
        listoffsetarray.to_typetracer()[
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
    array = ak.contents.NumpyArray(np.array(data))

    x1, x2, x3, x4, x5 = True, True, True, False, None
    mask = [x1, x2, x3, x4, x5]
    expected = [m if m is None else x for x, m in zip(data, mask) if m is not False]
    array2 = ak.highlevel.Array(mask, check_valid=True).layout

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
                        array2 = ak.highlevel.Array(mask, check_valid=True).layout
                        assert to_list(array[array2]) == expected
                        assert array.to_typetracer()[array2].form == array[array2].form


def test_bool_missing2():
    array = ak.highlevel.Array(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], check_valid=True
    ).layout
    array2 = ak.highlevel.Array([3, 6, None, None, -2, 6], check_valid=True).layout

    assert to_list(array[array2]) == [
        3.3,
        6.6,
        None,
        None,
        8.8,
        6.6,
    ]
    assert array.to_typetracer()[array2].form == array[array2].form

    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    regulararray = ak.contents.RegularArray(content, 4, zeros_length=0)

    assert to_list(regulararray) == [
        [0.0, 1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6, 7.7],
        [8.8, 9.9, 10.0, 11.1],
    ]

    array1 = ak.operations.from_iter([True, None, False, True], highlevel=False)

    assert to_list(regulararray[:, array1]) == [
        [0.0, None, 3.3],
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert regulararray.to_typetracer()[:, array1].form == regulararray[:, array1].form

    assert to_list(regulararray[1:, array1]) == [
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert (
        regulararray.to_typetracer()[1:, array1].form == regulararray[1:, array1].form
    )

    content = ak.contents.NumpyArray(
        np.array([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7], [8.8, 9.9, 10.0, 11.1]])
    )

    assert to_list(content[:, array1]) == [
        [0.0, None, 3.3],
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert content.to_typetracer()[:, array1].form == content[:, array1].form

    assert to_list(content[1:, array1]) == [[4.4, None, 7.7], [8.8, None, 11.1]]
    assert content.to_typetracer()[1:, array1].form == content[1:, array1].form

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 999])
    )
    offsets = ak.index.Index64(np.array([0, 4, 8, 12], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(listoffsetarray[:, array1]) == [
        [0.0, None, 3.3],
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert (
        listoffsetarray.to_typetracer()[:, array1].form
        == listoffsetarray[:, array1].form
    )

    assert to_list(listoffsetarray[1:, array1]) == [
        [4.4, None, 7.7],
        [8.8, None, 11.1],
    ]
    assert (
        listoffsetarray.to_typetracer()[1:, array1].form
        == listoffsetarray[1:, array1].form
    )


def test_records_missing():
    array = ak.highlevel.Array(
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
    ).layout
    array2 = ak.highlevel.Array([3, 1, None, 1, 7], check_valid=True).layout

    assert to_list(array[array2]) == [
        {"x": 3, "y": 3.3},
        {"x": 1, "y": 1.1},
        None,
        {"x": 1, "y": 1.1},
        {"x": 7, "y": 7.7},
    ]
    assert array.to_typetracer()[array2].form == array[array2].form

    array = ak.highlevel.Array(
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
    ).layout
    array2 = ak.highlevel.Array([1, None, 2, -1], check_valid=True).layout

    assert to_list(array[:, array2]) == [
        [{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [{"x": 5, "y": 5.5}, None, {"x": 6, "y": 6.6}, {"x": 9, "y": 9.9}],
    ]
    assert array.to_typetracer()[:, array2].form == array[:, array2].form

    array = ak.highlevel.Array(
        [
            {"x": [0, 1, 2, 3], "y": [0.0, 1.1, 2.2, 3.3]},
            {"x": [4, 5, 6, 7], "y": [4.4, 5.5, 6.6, 7.7]},
            {"x": [8, 9, 10, 11], "y": [8.8, 9.9, 10.0, 11.1]},
        ],
        check_valid=True,
    ).layout

    assert to_list(array[:, array2]) == [
        {"x": [1, None, 2, 3], "y": [1.1, None, 2.2, 3.3]},
        {"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]},
        {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]},
    ]
    assert array.to_typetracer()[:, array2].form == array[:, array2].form
    assert to_list(array[1:, array2]) == [
        {"x": [5, None, 6, 7], "y": [5.5, None, 6.6, 7.7]},
        {"x": [9, None, 10, 11], "y": [9.9, None, 10.0, 11.1]},
    ]
    assert array.to_typetracer()[1:, array2].form == array[1:, array2].form


def test_jagged():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array(
        [[0, -1], [], [-1, 0], [-1], [1, 1, -2, 0]], check_valid=True
    ).layout

    assert to_list(array[array2]) == [
        [1.1, 3.3],
        [],
        [5.5, 4.4],
        [6.6],
        [8.8, 8.8, 8.8, 7.7],
    ]
    assert array.to_typetracer()[array2].form == array[array2].form


def test_double_jagged():
    array = ak.highlevel.Array(
        [[[0, 1, 2, 3], [4, 5]], [[6, 7, 8], [9, 10, 11, 12, 13]]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array(
        [[[2, 1, 0], [-1]], [[-1, -2, -3], [2, 1, 1, 3]]], check_valid=True
    ).layout

    assert to_list(array[array2]) == [
        [[2, 1, 0], [5]],
        [[8, 7, 6], [11, 10, 10, 12]],
    ]
    assert array.to_typetracer()[array2].form == array[array2].form

    content = ak.operations.from_iter(
        [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12, 13]], highlevel=False
    )
    regulararray = ak.contents.RegularArray(content, 2, zeros_length=0)

    array1 = ak.highlevel.Array([[2, 1, 0], [-1]], check_valid=True).layout

    assert to_list(regulararray[:, array1]) == [[[2, 1, 0], [5]], [[8, 7, 6], [13]]]
    assert regulararray.to_typetracer()[:, array1].form == regulararray[:, array1].form
    assert to_list(regulararray[1:, array1]) == [[[8, 7, 6], [13]]]
    assert (
        regulararray.to_typetracer()[1:, array1].form == regulararray[1:, array1].form
    )

    offsets = ak.index.Index64(np.array([0, 2, 4], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    assert to_list(listoffsetarray[:, array1]) == [
        [[2, 1, 0], [5]],
        [[8, 7, 6], [13]],
    ]
    assert (
        listoffsetarray.to_typetracer()[:, array1].form
        == listoffsetarray[:, array1].form
    )
    assert to_list(listoffsetarray[1:, array1]) == [[[8, 7, 6], [13]]]
    assert (
        listoffsetarray.to_typetracer()[1:, array1].form
        == listoffsetarray[1:, array1].form
    )


def test_masked_jagged():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    array1 = ak.highlevel.Array(
        [[-1, -2], None, [], None, [-2, 0]], check_valid=True
    ).layout

    assert to_list(array[array1]) == [[3.3, 2.2], None, [], None, [8.8, 7.7]]
    assert array.to_typetracer()[array1].form == array[array1].form


def test_jagged_masked():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    array1 = ak.highlevel.Array(
        [[-1, None], [], [None, 0], [None], [1]], check_valid=True
    ).layout

    assert to_list(array[array1]) == [[3.3, None], [], [None, 4.4], [None], [8.8]]
    assert array.to_typetracer()[array1].form == array[array1].form


def test_regular_regular():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5))
    regulararray1 = ak.contents.RegularArray(content, 5, zeros_length=0)
    regulararray2 = ak.contents.RegularArray(regulararray1, 3, zeros_length=0)

    array1 = ak.highlevel.Array(
        [[[2], [1, -2], [-1, 2, 0]], [[-3], [-4, 3], [-5, -3, 4]]],
        check_valid=True,
    ).layout
    array2 = ak.highlevel.Array(
        [[[2], [1, -2], [-1, None, 0]], [[-3], [-4, 3], [-5, None, 4]]],
        check_valid=True,
    ).layout

    assert to_list(regulararray2[array1]) == [
        [[2], [6, 8], [14, 12, 10]],
        [[17], [21, 23], [25, 27, 29]],
    ]
    assert regulararray2.to_typetracer()[array1].form == regulararray2[array1].form

    assert to_list(regulararray2[array2]) == [
        [[2], [6, 8], [14, None, 10]],
        [[17], [21, 23], [25, None, 29]],
    ]
    assert regulararray2.to_typetracer()[array2].form == regulararray2[array2].form


def test_masked_of_jagged_of_whatever():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5))
    regulararray1 = ak.contents.RegularArray(content, 5, zeros_length=0)
    regulararray2 = ak.contents.RegularArray(regulararray1, 3, zeros_length=0)

    array1 = ak.highlevel.Array(
        [[[2], None, [-1, 2, 0]], [[-3], None, [-5, -3, 4]]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array(
        [[[2], None, [-1, None, 0]], [[-3], None, [-5, None, 4]]],
        check_valid=True,
    ).layout

    assert to_list(regulararray2[array1]) == [
        [[2], None, [14, 12, 10]],
        [[17], None, [25, 27, 29]],
    ]
    assert regulararray2.to_typetracer()[array1].form == regulararray2[array1].form

    assert to_list(regulararray2[array2]) == [
        [[2], None, [14, None, 10]],
        [[17], None, [25, None, 29]],
    ]
    assert regulararray2.to_typetracer()[array2].form == regulararray2[array2].form


def test_emptyarray():
    content = ak.contents.EmptyArray()
    offsets = ak.index.Index64(np.array([0, 0, 0, 0, 0], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    array1 = ak.highlevel.Array([[], [], [], []], check_valid=True).layout
    array2 = ak.highlevel.Array([[], [None], [], []], check_valid=True).layout
    array3 = ak.highlevel.Array([[], [], None, []], check_valid=True).layout
    array4 = ak.highlevel.Array([[], [None], None, []], check_valid=True).layout
    array5 = ak.highlevel.Array([[], [0], [], []], check_valid=True).layout

    assert to_list(listoffsetarray) == [[], [], [], []]

    assert to_list(listoffsetarray[array1]) == [[], [], [], []]
    assert listoffsetarray.to_typetracer()[array1].form == listoffsetarray[array1].form

    assert to_list(listoffsetarray[array2]) == [[], [None], [], []]
    assert listoffsetarray.to_typetracer()[array2].form == listoffsetarray[array2].form
    assert to_list(listoffsetarray[array3]) == [[], [], None, []]
    assert listoffsetarray.to_typetracer()[array3].form == listoffsetarray[array3].form
    assert to_list(listoffsetarray[array4]) == [[], [None], None, []]
    assert listoffsetarray.to_typetracer()[array4].form == listoffsetarray[array4].form

    with pytest.raises(IndexError):
        listoffsetarray[array5]


def test_numpyarray():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True
    ).layout
    array2 = ak.highlevel.Array([[[], [], []], [], [[], []]], check_valid=True).layout

    with pytest.raises(IndexError):
        array[array2]


def test_record():
    array = ak.highlevel.Array(
        [
            {"x": [0, 1, 2], "y": [0.0, 1.1, 2.2, 3.3]},
            {"x": [3, 4, 5, 6], "y": [4.4, 5.5]},
            {"x": [7, 8], "y": [6.6, 7.7, 8.8, 9.9]},
        ],
        check_valid=True,
    ).layout
    array2 = ak.highlevel.Array([[-1, 1], [0, 0, 1], [-1, -2]], check_valid=True).layout
    array3 = ak.highlevel.Array(
        [[-1, 1], [0, 0, None, 1], [-1, -2]], check_valid=True
    ).layout
    array4 = ak.highlevel.Array([[-1, 1], None, [-1, -2]], check_valid=True).layout

    assert to_list(array[array2]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        {"x": [3, 3, 4], "y": [4.4, 4.4, 5.5]},
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert array.to_typetracer()[array2].form == array[array2].form
    assert to_list(array[array3]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        {"x": [3, 3, None, 4], "y": [4.4, 4.4, None, 5.5]},
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert array.to_typetracer()[array3].form == array[array3].form
    assert to_list(array[array4]) == [
        {"x": [2, 1], "y": [3.3, 1.1]},
        None,
        {"x": [8, 7], "y": [9.9, 8.8]},
    ]
    assert array.to_typetracer()[array4].form == array[array4].form


def test_indexedarray():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.index.Index64(np.array([3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, array)

    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
    ]

    array1 = ak.highlevel.Array([[0, -1], [0], [], [1, 1]], check_valid=True).layout
    assert to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], [], [1.1, 1.1]]
    assert indexedarray.to_typetracer()[array1].form == indexedarray[array1].form

    array1 = ak.highlevel.Array(
        [[0, -1], [0], [None], [1, None, 1]], check_valid=True
    ).layout

    assert to_list(indexedarray[array1]) == [
        [6.6, 9.9],
        [5.5],
        [None],
        [1.1, None, 1.1],
    ]
    assert indexedarray.to_typetracer()[array1].form == indexedarray[array1].form

    array1 = ak.highlevel.Array([[0, -1], [0], None, [1, 1]], check_valid=True).layout

    assert to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], None, [1.1, 1.1]]
    assert indexedarray.to_typetracer()[array1].form == indexedarray[array1].form

    array1 = ak.highlevel.Array([[0, -1], [0], None, [None]], check_valid=True).layout

    assert to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], None, [None]]
    assert indexedarray.to_typetracer()[array1].form == indexedarray[array1].form

    index = ak.index.Index64(np.array([3, 2, 1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, array)

    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
    ]

    assert to_list(
        indexedarray[ak.highlevel.Array([[0, -1], [0], [], [1, 1]], check_valid=True)]
    ) == [[6.6, 9.9], [5.5], [], [1.1, 1.1]]
    assert (
        indexedarray.to_typetracer()[
            ak.highlevel.Array([[0, -1], [0], [], [1, 1]])
        ].form
        == indexedarray[ak.highlevel.Array([[0, -1], [0], [], [1, 1]])].form
    )

    array1 = ak.highlevel.Array(
        [[0, -1], [0], [None], [1, None, 1]], check_valid=True
    ).layout

    assert to_list(indexedarray[array1]) == [
        [6.6, 9.9],
        [5.5],
        [None],
        [1.1, None, 1.1],
    ]
    assert indexedarray.to_typetracer()[array1].form == indexedarray[array1].form

    array1 = ak.highlevel.Array([[0, -1], [0], None, []], check_valid=True).layout

    assert to_list(indexedarray[array1]) == [[6.6, 9.9], [5.5], None, []]
    assert indexedarray.to_typetracer()[array1].form == indexedarray[array1].form

    array1 = ak.highlevel.Array(
        [[0, -1], [0], None, [1, None, 1]], check_valid=True
    ).layout

    assert to_list(indexedarray[array1]) == [
        [6.6, 9.9],
        [5.5],
        None,
        [1.1, None, 1.1],
    ]
    assert indexedarray.to_typetracer()[array1].form == indexedarray[array1].form


def test_indexedarray2():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.index.Index64(np.array([3, 2, -1, 0], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, array)
    array = ak.highlevel.Array([[0, -1], [0], None, [1, 1]]).layout

    assert to_list(indexedarray) == [
        [6.6, 7.7, 8.8, 9.9],
        [5.5],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert to_list(indexedarray[array]) == [
        [6.6, 9.9],
        [5.5],
        None,
        [1.1, 1.1],
    ]
    assert indexedarray.to_typetracer()[array].form == indexedarray[array].form


def test_indexedarray2b():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.index.Index64(np.array([0, -1, 2, 3], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, array)
    array = ak.highlevel.Array([[1, 1], None, [0], [0, -1]]).layout

    assert to_list(indexedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(indexedarray[array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert indexedarray.to_typetracer()[array].form == indexedarray[array].form


def test_bytemaskedarray2b():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.Index8(np.array([0, 1, 0, 0], dtype=np.int8))
    maskedarray = ak.contents.ByteMaskedArray(mask, array, valid_when=False)
    array = ak.highlevel.Array([[1, 1], None, [0], [0, -1]]).layout

    assert to_list(maskedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(maskedarray[array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert maskedarray.to_typetracer()[array].form == maskedarray[array].form


def test_bitmaskedarray2b():
    array = ak.operations.from_iter(
        [[0.0, 1.1, 2.2], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    mask = ak.index.IndexU8(np.array([66], dtype=np.uint8))
    maskedarray = ak.contents.BitMaskedArray(
        mask, array, valid_when=False, length=4, lsb_order=True
    )  # lsb_order is irrelevant in this example
    array = ak.highlevel.Array([[1, 1], None, [0], [0, -1]]).layout

    assert to_list(maskedarray) == [
        [0.0, 1.1, 2.2],
        None,
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert to_list(maskedarray[array]) == [
        [1.1, 1.1],
        None,
        [5.5],
        [6.6, 9.9],
    ]
    assert maskedarray.to_typetracer()[array].form == maskedarray[array].form


def test_indexedarray3():
    array = ak.highlevel.Array([0.0, 1.1, 2.2, None, 4.4, None, None, 7.7]).layout
    assert to_list(array[ak.highlevel.Array([4, 3, 2])]) == [4.4, None, 2.2]
    assert to_list(array[ak.highlevel.Array([4, 3, 2, None, 1])]) == [
        4.4,
        None,
        2.2,
        None,
        1.1,
    ]

    array = ak.highlevel.Array([[0.0, 1.1, None, 2.2], [3.3, None, 4.4], [5.5]]).layout
    array2 = ak.highlevel.Array([[3, 2, 2, 1], [1, 2], []]).layout

    assert to_list(array[array2]) == [
        [2.2, None, None, 1.1],
        [None, 4.4],
        [],
    ]
    assert array.to_typetracer()[array2].form == array[array2].form

    array = ak.highlevel.Array([[0.0, 1.1, 2.2], [3.3, 4.4], None, [5.5]]).layout
    array2 = ak.highlevel.Array([3, 2, 1]).layout
    array3 = ak.highlevel.Array([3, 2, 1, None, 0]).layout
    array4 = ak.highlevel.Array([[2, 1, 1, 0], [1], None, [0]]).layout
    array5 = ak.highlevel.Array([[2, 1, 1, 0], None, [1], [0]]).layout
    array6 = ak.highlevel.Array([[2, 1, 1, 0], None, [1], [0], None]).layout

    assert to_list(array[array2]) == [[5.5], None, [3.3, 4.4]]
    assert array.to_typetracer()[array2].form == array[array2].form
    assert to_list(array[array3]) == [
        [5.5],
        None,
        [3.3, 4.4],
        None,
        [0.0, 1.1, 2.2],
    ]
    assert array.to_typetracer()[array3].form == array[array3].form

    assert (to_list(array[array4])) == [
        [2.2, 1.1, 1.1, 0.0],
        [4.4],
        None,
        [5.5],
    ]
    assert array.to_typetracer()[array4].form == array[array4].form

    assert to_list(array[array5]) == [
        [2.2, 1.1, 1.1, 0],
        None,
        None,
        [5.5],
    ]
    assert array.to_typetracer()[array5].form == array[array5].form
    with pytest.raises(IndexError):
        array[array6]


def test_sequential():
    array = ak.highlevel.Array(
        np.arange(2 * 3 * 5).reshape(2, 3, 5).tolist(), check_valid=True
    ).layout
    array2 = ak.highlevel.Array([[2, 1, 0], [2, 1, 0]], check_valid=True).layout

    assert to_list(array[array2]) == [
        [[10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
        [[25, 26, 27, 28, 29], [20, 21, 22, 23, 24], [15, 16, 17, 18, 19]],
    ]
    assert array.to_typetracer()[array2].form == array[array2].form
    assert to_list(array[array2, :2]) == [
        [[10, 11], [5, 6], [0, 1]],
        [[25, 26], [20, 21], [15, 16]],
    ]
    assert array.to_typetracer()[array2, :2].form == array[array2, :2].form


def test_union():
    one = ak.operations.from_iter(
        [["1.1", "2.2", "3.3"], [], ["4.4", "5.5"]], highlevel=False
    )
    two = ak.operations.from_iter(
        [[6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]], highlevel=False
    )
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray(tags, index, [one, two])

    assert to_list(unionarray) == [
        ["1.1", "2.2", "3.3"],
        [],
        ["4.4", "5.5"],
        [6.6],
        [7.7, 8.8],
        [],
        [9.9, 10.0, 11.1, 12.2],
    ]


def test_union_2():
    one = ak.operations.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    two = ak.operations.from_iter(
        [[6.6], [7.7, 8.8], [], [9.9, 10.0, 11.1, 12.2]], highlevel=False
    )
    tags = ak.index.Index8(np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 1, 2, 0, 1, 2, 3], dtype=np.int64))
    unionarray = ak.contents.UnionArray.simplified(tags, index, [one, two])
    array = ak.highlevel.Array(
        [[0, -1], [], [1, 1], [], [-1], [], [1, -2, -1]], check_valid=True
    ).layout

    assert to_list(unionarray[array]) == [
        [1.1, 3.3],
        [],
        [5.5, 5.5],
        [],
        [8.8],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert unionarray.to_typetracer()[array].form == unionarray[array].form


def test_jagged_mask():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]], check_valid=True
    ).layout
    assert to_list(
        array[[[True, True, True], [], [True, True], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.to_typetracer()[
            [[True, True, True], [], [True, True], [True], [True, True, True]]
        ].form
        == array[
            [[True, True, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        array[[[False, True, True], [], [True, True], [True], [True, True, True]]]
    ) == [[2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.to_typetracer()[
            [[False, True, True], [], [True, True], [True], [True, True, True]]
        ].form
        == array[
            [[False, True, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        array[[[True, False, True], [], [True, True], [True], [True, True, True]]]
    ) == [[1.1, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.to_typetracer()[
            [[True, False, True], [], [True, True], [True], [True, True, True]]
        ].form
        == array[
            [[True, False, True], [], [True, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        array[[[True, True, True], [], [False, True], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [5.5], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.to_typetracer()[
            [[True, True, True], [], [False, True], [True], [True, True, True]]
        ].form
        == array[
            [[True, True, True], [], [False, True], [True], [True, True, True]]
        ].form
    )
    assert to_list(
        array[[[True, True, True], [], [False, False], [True], [True, True, True]]]
    ) == [[1.1, 2.2, 3.3], [], [], [6.6], [7.7, 8.8, 9.9]]
    assert (
        array.to_typetracer()[
            [[True, True, True], [], [False, False], [True], [True, True, True]]
        ].form
        == array[
            [[True, True, True], [], [False, False], [True], [True, True, True]]
        ].form
    )


def test_jagged_missing_mask():
    array = ak.highlevel.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True
    ).layout

    assert to_list(array[[[True, True, True], [], [True, True]]]) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.to_typetracer()[[[True, True, True], [], [True, True]]].form
        == array[[[True, True, True], [], [True, True]]].form
    )
    assert to_list(array[[[True, False, True], [], [True, True]]]) == [
        [1.1, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.to_typetracer()[[[True, False, True], [], [True, True]]].form
        == array[[[True, False, True], [], [True, True]]].form
    )
    assert to_list(array[[[True, True, False], [], [False, None]]]) == [
        [1.1, 2.2],
        [],
        [None],
    ]
    assert (
        array.to_typetracer()[[[True, True, False], [], [False, None]]].form
        == array[[[True, True, False], [], [False, None]]].form
    )
    assert to_list(array[[[True, True, False], [], [True, None]]]) == [
        [1.1, 2.2],
        [],
        [4.4, None],
    ]
    assert (
        array.to_typetracer()[[[True, True, False], [], [True, None]]].form
        == array[[[True, True, False], [], [True, None]]].form
    )

    assert to_list(array[[[True, None, True], [], [True, True]]]) == [
        [1.1, None, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.to_typetracer()[[[True, None, True], [], [True, True]]].form
        == array[[[True, None, True], [], [True, True]]].form
    )
    assert to_list(array[[[True, None, False], [], [True, True]]]) == [
        [1.1, None],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.to_typetracer()[[[True, None, False], [], [True, True]]].form
        == array[[[True, None, False], [], [True, True]]].form
    )

    assert to_list(array[[[False, None, False], [], [True, True]]]) == [
        [None],
        [],
        [4.4, 5.5],
    ]
    assert (
        array.to_typetracer()[[[False, None, False], [], [True, True]]].form
        == array[[[False, None, False], [], [True, True]]].form
    )
    assert to_list(array[[[True, True, False], [], [False, True]]]) == [
        [1.1, 2.2],
        [],
        [5.5],
    ]
    assert (
        array.to_typetracer()[[[True, True, False], [], [False, True]]].form
        == array[[[True, True, False], [], [False, True]]].form
    )
    assert to_list(array[[[True, True, None], [], [False, True]]]) == [
        [1.1, 2.2, None],
        [],
        [5.5],
    ]
    assert (
        array.to_typetracer()[[[True, True, None], [], [False, True]]].form
        == array[[[True, True, None], [], [False, True]]].form
    )
    assert to_list(array[[[True, True, False], [None], [False, True]]]) == [
        [1.1, 2.2],
        [None],
        [5.5],
    ]
    assert (
        array.to_typetracer()[[[True, True, False], [None], [False, True]]].form
        == array[[[True, True, False], [None], [False, True]]].form
    )


def test_array_boolean_to_int():
    a = ak.operations.from_iter(
        [[True, True, True], [], [True, True], [True], [True, True, True, True]],
        highlevel=False,
    )
    b = ak._slicing.normalise_item_bool_to_int(a)
    assert to_list(b) == [[0, 1, 2], [], [0, 1], [0], [0, 1, 2, 3]]

    a = ak.operations.from_iter(
        [
            [True, True, False],
            [],
            [True, False],
            [False],
            [True, True, True, False],
        ],
        highlevel=False,
    )
    b = ak._slicing.normalise_item_bool_to_int(a)
    assert to_list(b) == [[0, 1], [], [0], [], [0, 1, 2]]

    a = ak.operations.from_iter(
        [
            [False, True, True],
            [],
            [False, True],
            [False],
            [False, True, True, True],
        ],
        highlevel=False,
    )
    b = ak._slicing.normalise_item_bool_to_int(a)
    assert to_list(b) == [[1, 2], [], [1], [], [1, 2, 3]]

    a = ak.operations.from_iter(
        [[True, True, None], [], [True, None], [None], [True, True, True, None]],
        highlevel=False,
    )
    b = ak._slicing.normalise_item_bool_to_int(a)
    assert to_list(b) == [[0, 1, None], [], [0, None], [None], [0, 1, 2, None]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(6).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = ak.operations.from_iter(
        [[None, True, True], [], [None, True], [None], [None, True, True, True]],
        highlevel=False,
    )
    b = ak._slicing.normalise_item_bool_to_int(a)
    assert to_list(b) == [[None, 1, 2], [], [None, 1], [None], [None, 1, 2, 3]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(6).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = ak.operations.from_iter(
        [[False, True, None], [], [False, None], [None], [False, True, True, None]],
        highlevel=False,
    )
    b = ak._slicing.normalise_item_bool_to_int(a)
    assert to_list(b) == [[1, None], [], [None], [None], [1, 2, None]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(3).tolist()  # kernels expect nonnegative entries to be arange
    )

    a = ak.operations.from_iter(
        [[None, True, False], [], [None, False], [None], [None, True, True, False]],
        highlevel=False,
    )
    b = ak._slicing.normalise_item_bool_to_int(a)
    assert to_list(b) == [[None, 1], [], [None], [None], [None, 1, 2]]
    assert (
        b.content.index.data[b.content.index.data >= 0].tolist()
        == np.arange(3).tolist()  # kernels expect nonnegative entries to be arange
    )
