# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list

content = ak.contents.NumpyArray(
    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
)
starts1 = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
stops1 = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
offsets1 = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9]))
starts2 = ak.index.Index64(np.array([0, 2, 3, 3]))
stops2 = ak.index.Index64(np.array([2, 3, 3, 5]))
offsets2 = ak.index.Index64(np.array([0, 2, 3, 3, 5]))


def test_listarray_basic():
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array2 = ak.contents.ListArray(starts2, stops2, array1)

    assert to_list(array1) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert to_list(array1[2]) == [4.4, 5.5]
    assert array1.to_typetracer()[2].form == array1[2].form
    assert to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert array1.to_typetracer()[1:-1].form == array1[1:-1].form
    assert to_list(array2) == [
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert to_list(array2[1]) == [[4.4, 5.5]]
    assert array2.to_typetracer()[1].form == array2[1].form
    assert to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert array2.to_typetracer()[1:-1].form == array2[1:-1].form


def test_listoffsetarray_basic():
    array1 = ak.contents.ListOffsetArray(offsets1, content)
    array2 = ak.contents.ListOffsetArray(offsets2, array1)

    assert to_list(array1) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert to_list(array1[2]) == [4.4, 5.5]
    assert array1.to_typetracer()[2].form == array1[2].form
    assert to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert array1.to_typetracer()[1:-1].form == array1[1:-1].form
    assert to_list(array2) == [
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [],
        [[6.6], [7.7, 8.8, 9.9]],
    ]
    assert to_list(array2[1]) == [[4.4, 5.5]]
    assert array2.to_typetracer()[1].form == array2[1].form
    assert to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert array2.to_typetracer()[1:-1].form == array2[1:-1].form


def test_listarray_at():
    array1 = ak.contents.ListArray(starts1, stops1, content)

    assert to_list(array1[2]) == [4.4, 5.5]
    assert array1.to_typetracer()[2].form == array1[2].form
    assert to_list(
        array1[
            2,
        ]
    ) == [4.4, 5.5]
    assert to_list(array1[2, 1:]) == [5.5]
    assert array1.to_typetracer()[2, 1:].form == array1[2, 1:].form
    assert to_list(array1[2:, 0]) == [4.4, 6.6, 7.7]
    assert array1.to_typetracer()[2:, 0].form == array1[2:, 0].form
    assert to_list(array1[2:, -1]) == [5.5, 6.6, 9.9]
    assert array1.to_typetracer()[2:, -1].form == array1[2:, -1].form


def test_listoffsetarray_at():
    array1 = ak.contents.ListOffsetArray(offsets1, content)

    assert to_list(
        array1[
            2,
        ]
    ) == [4.4, 5.5]
    assert to_list(array1[2, 1:]) == [5.5]
    assert array1.to_typetracer()[2, 1:].form == array1[2, 1:].form
    assert to_list(array1[2:, 0]) == [4.4, 6.6, 7.7]
    assert array1.to_typetracer()[2:, 0].form == array1[2:, 0].form
    assert to_list(array1[2:, -1]) == [5.5, 6.6, 9.9]
    assert array1.to_typetracer()[2:, -1].form == array1[2:, -1].form


def test_listarray_slice():
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array2 = ak.contents.ListArray(starts2, stops2, array1)

    assert to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert array1.to_typetracer()[1:-1].form == array1[1:-1].form
    assert to_list(
        array1[
            1:-1,
        ]
    ) == [[], [4.4, 5.5], [6.6]]
    assert (
        array1.to_typetracer()[
            1:-1,
        ].form
        == array1[
            1:-1,
        ].form
    )
    assert to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert array2.to_typetracer()[1:-1].form == array2[1:-1].form
    assert to_list(
        array2[
            1:-1,
        ]
    ) == [[[4.4, 5.5]], []]
    assert (
        array2.to_typetracer()[
            1:-1,
        ].form
        == array2[
            1:-1,
        ].form
    )


def test_listoffsetarray_slice():
    array1 = ak.contents.ListOffsetArray(offsets1, content)
    array2 = ak.contents.ListOffsetArray(offsets2, array1)

    assert to_list(array1[1:-1]) == [[], [4.4, 5.5], [6.6]]
    assert array1.to_typetracer()[1:-1].form == array1[1:-1].form
    assert to_list(
        array1[
            1:-1,
        ]
    ) == [[], [4.4, 5.5], [6.6]]
    assert (
        array1.to_typetracer()[
            1:-1,
        ].form
        == array1[
            1:-1,
        ].form
    )
    assert to_list(array2[1:-1]) == [[[4.4, 5.5]], []]
    assert array2.to_typetracer()[1:-1].form == array2[1:-1].form
    assert to_list(
        array2[
            1:-1,
        ]
    ) == [[[4.4, 5.5]], []]
    assert (
        array2.to_typetracer()[
            1:-1,
        ].form
        == array2[
            1:-1,
        ].form
    )


def test_listarray_slice_slice():
    array1 = ak.contents.ListArray(starts1, stops1, content)

    assert to_list(array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert array1.to_typetracer()[2:].form == array1[2:].form
    assert to_list(array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert array1.to_typetracer()[2:, 1:].form == array1[2:, 1:].form
    assert to_list(array1[2:, :-1]) == [[4.4], [], [7.7, 8.8]]
    assert array1.to_typetracer()[2:, :-1].form == array1[2:, :-1].form


def test_listoffsetarray_slice_slice():
    array1 = ak.contents.ListOffsetArray(offsets1, content)

    assert to_list(array1[2:]) == [[4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert array1.to_typetracer()[2:].form == array1[2:].form
    assert to_list(array1[2:, 1:]) == [[5.5], [], [8.8, 9.9]]
    assert array1.to_typetracer()[2:, 1:].form == array1[2:, 1:].form
    assert to_list(array1[2:, :-1]) == [[4.4], [], [7.7, 8.8]]
    assert array1.to_typetracer()[2:, :-1].form == array1[2:, :-1].form


def test_listarray_ellipsis():
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array2 = ak.contents.ListArray(starts2, stops2, array1)

    assert to_list(array1[Ellipsis, 1:]) == [
        [2.2, 3.3],
        [],
        [5.5],
        [],
        [8.8, 9.9],
    ]
    assert array1.to_typetracer()[Ellipsis, 1:].form == array1[Ellipsis, 1:].form
    assert to_list(array2[Ellipsis, 1:]) == [
        [[2.2, 3.3], []],
        [[5.5]],
        [],
        [[], [8.8, 9.9]],
    ]
    assert array2.to_typetracer()[Ellipsis, 1:].form == array2[Ellipsis, 1:].form


def test_listoffsetarray_ellipsis():
    array1 = ak.contents.ListOffsetArray(offsets1, content)
    array2 = ak.contents.ListOffsetArray(offsets2, array1)

    assert to_list(array1[Ellipsis, 1:]) == [
        [2.2, 3.3],
        [],
        [5.5],
        [],
        [8.8, 9.9],
    ]
    assert array1.to_typetracer()[Ellipsis, 1:].form == array1[Ellipsis, 1:].form
    assert to_list(array2[Ellipsis, 1:]) == [
        [[2.2, 3.3], []],
        [[5.5]],
        [],
        [[], [8.8, 9.9]],
    ]
    assert array2.to_typetracer()[Ellipsis, 1:].form == array2[Ellipsis, 1:].form


def test_listarray_array_slice():
    array1 = ak.contents.ListArray(starts1, stops1, content)
    array2 = ak.contents.ListArray(starts2, stops2, array1)

    assert to_list(array2[[0, 0, 1, 1, 1, 0]]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        array2.to_typetracer()[[0, 0, 1, 1, 1, 0]].form
        == array2[[0, 0, 1, 1, 1, 0]].form
    )
    assert to_list(array2[[0, 0, 1, 1, 1, 0], :]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :].form
        == array2[[0, 0, 1, 1, 1, 0], :].form
    )
    assert to_list(array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [
        [[2.2, 3.3], []],
        [[2.2, 3.3], []],
        [[5.5]],
        [[5.5]],
        [[5.5]],
        [[2.2, 3.3], []],
    ]
    assert (
        array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :, 1:].form
        == array2[[0, 0, 1, 1, 1, 0], :, 1:].form
    )


def test_listoffsetarray_array_slice():
    array1 = ak.contents.ListOffsetArray(offsets1, content)
    array2 = ak.contents.ListOffsetArray(offsets2, array1)

    assert to_list(array2[[0, 0, 1, 1, 1, 0]]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        array2.to_typetracer()[[0, 0, 1, 1, 1, 0]].form
        == array2[[0, 0, 1, 1, 1, 0]].form
    )
    assert to_list(array2[[0, 0, 1, 1, 1, 0], :]) == [
        [[1.1, 2.2, 3.3], []],
        [[1.1, 2.2, 3.3], []],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[4.4, 5.5]],
        [[1.1, 2.2, 3.3], []],
    ]
    assert (
        array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :].form
        == array2[[0, 0, 1, 1, 1, 0], :].form
    )
    assert to_list(array2[[0, 0, 1, 1, 1, 0], :, 1:]) == [
        [[2.2, 3.3], []],
        [[2.2, 3.3], []],
        [[5.5]],
        [[5.5]],
        [[5.5]],
        [[2.2, 3.3], []],
    ]
    assert (
        array2.to_typetracer()[[0, 0, 1, 1, 1, 0], :, 1:].form
        == array2[[0, 0, 1, 1, 1, 0], :, 1:].form
    )


def test_listarray_array():
    array1 = ak.contents.ListArray(starts1, stops1, content)

    assert to_list(array1[np.array([2, 0, 0, 1, -1])]) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [7.7, 8.8, 9.9],
    ]
    assert (
        array1.to_typetracer()[np.array([2, 0, 0, 1, -1])].form
        == array1[np.array([2, 0, 0, 1, -1])].form
    )

    content_deep = ak.contents.NumpyArray(
        np.array(
            [
                [0, 0],
                [1, 10],
                [2, 20],
                [3, 30],
                [4, 40],
                [5, 50],
                [6, 60],
                [7, 70],
                [8, 80],
            ]
        )
    )
    starts1_deep = ak.index.Index64(np.array([0, 3, 6]))
    stops1_deep = ak.index.Index64(np.array([3, 6, 9]))
    array1_deep = ak.contents.ListArray(starts1_deep, stops1_deep, content_deep)

    assert to_list(array1_deep) == [
        [[0, 0], [1, 10], [2, 20]],
        [[3, 30], [4, 40], [5, 50]],
        [[6, 60], [7, 70], [8, 80]],
    ]
    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1]))
    assert np.array(
        [
            [[0, 0], [1, 10], [2, 20]],
            [[3, 30], [4, 40], [5, 50]],
            [[6, 60], [7, 70], [8, 80]],
        ]
    )[s].tolist() == to_list(array1_deep[s])

    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), slice(1, None))
    assert np.array(
        [
            [[0, 0], [1, 10], [2, 20]],
            [[3, 30], [4, 40], [5, 50]],
            [[6, 60], [7, 70], [8, 80]],
        ]
    )[s].tolist() == to_list(array1_deep[s])


def test_listoffsetarray_array():
    array1 = ak.contents.ListOffsetArray(offsets1, content)

    assert to_list(array1[np.array([2, 0, 0, 1, -1])]) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        [],
        [7.7, 8.8, 9.9],
    ]
    assert (
        array1.to_typetracer()[np.array([2, 0, 0, 1, -1])].form
        == array1[np.array([2, 0, 0, 1, -1])].form
    )

    content_deep = ak.contents.NumpyArray(
        np.array(
            [
                [0, 0],
                [1, 10],
                [2, 20],
                [3, 30],
                [4, 40],
                [5, 50],
                [6, 60],
                [7, 70],
                [8, 80],
            ]
        )
    )
    starts1_deep = ak.index.Index64(np.array([0, 3, 6]))
    stops1_deep = ak.index.Index64(np.array([3, 6, 9]))
    array1_deep = ak.contents.ListArray(starts1_deep, stops1_deep, content_deep)

    assert to_list(array1_deep) == [
        [[0, 0], [1, 10], [2, 20]],
        [[3, 30], [4, 40], [5, 50]],
        [[6, 60], [7, 70], [8, 80]],
    ]
    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1]))
    assert np.array(
        [
            [[0, 0], [1, 10], [2, 20]],
            [[3, 30], [4, 40], [5, 50]],
            [[6, 60], [7, 70], [8, 80]],
        ]
    )[s].tolist() == to_list(array1_deep[s])

    s = (np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0]), slice(1, None))
    assert np.array(
        [
            [[0, 0], [1, 10], [2, 20]],
            [[3, 30], [4, 40], [5, 50]],
            [[6, 60], [7, 70], [8, 80]],
        ]
    )[s].tolist() == to_list(array1_deep[s])


def test_listarray_listoffsetarray_array():
    array1 = ak.contents.ListArray(starts1, stops1, content)

    assert to_list(array1[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])]) == [
        5.5,
        2.2,
        1.1,
        7.7,
    ]
    assert (
        array1.to_typetracer()[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])].form
        == array1[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])].form
    )

    array1 = ak.contents.ListOffsetArray(offsets1, content)

    assert to_list(array1[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])]) == [
        5.5,
        2.2,
        1.1,
        7.7,
    ]
    assert (
        array1.to_typetracer()[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])].form
        == array1[np.array([2, 0, 0, -1]), np.array([1, 1, 0, 0])].form
    )
