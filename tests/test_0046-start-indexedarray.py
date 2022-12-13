# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_basic():
    content = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int32)
    index = ak.index.Index32(ind)
    array = ak.contents.IndexedArray(index, content)

    assert to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.uint32)
    index = ak.index.IndexU32(ind)
    array = ak.contents.IndexedArray(index, content)
    assert to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int64)
    index = ak.index.Index64(ind)
    array = ak.contents.IndexedArray(index, content)
    assert to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int32)
    index = ak.index.Index32(ind)
    array = ak.contents.IndexedOptionArray(index, content)
    assert to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int64)
    index = ak.index.Index64(ind)
    array = ak.contents.IndexedOptionArray(index, content)
    assert to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]


def test_type():
    content = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.index.Index32(np.array([2, 2, 0, 3, 4], dtype=np.int32))
    array = ak.contents.IndexedArray(index, content)
    assert ak.operations.type(array) == ak.types.NumpyType("float64")
    array = ak.contents.IndexedOptionArray(index, content)
    assert ak.operations.type(array) == ak.types.OptionType(
        ak.types.NumpyType("float64")
    )


def test_null():
    content = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.index.Index64(np.array([2, 2, 0, -1, 4], dtype=np.int64))
    array = ak.contents.IndexedOptionArray(index, content)

    assert to_list(array) == [2.2, 2.2, 0.0, None, 4.4]


def test_carry():
    content = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.index.Index64(np.array([2, 2, 0, 3, 4], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)
    offsets = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, indexedarray)

    assert to_list(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, 4.4]]
    assert to_list(listoffsetarray[::-1]) == [[3.3, 4.4], [], [2.2, 2.2, 0.0]]
    assert listoffsetarray.to_typetracer()[::-1].form == listoffsetarray[::-1].form
    assert to_list(listoffsetarray[[2, 0]]) == [[3.3, 4.4], [2.2, 2.2, 0.0]]
    assert listoffsetarray.to_typetracer()[[2, 0]].form == listoffsetarray[[2, 0]].form
    assert to_list(listoffsetarray[[2, 0], 1]) == [4.4, 2.2]  # invokes carry
    assert (
        listoffsetarray.to_typetracer()[[2, 0], 1].form
        == listoffsetarray[[2, 0], 1].form
    )
    assert to_list(listoffsetarray[2:, 1]) == [4.4]  # invokes carry
    assert listoffsetarray.to_typetracer()[2:, 1].form == listoffsetarray[2:, 1].form

    index = ak.index.Index64(np.array([2, 2, 0, 3, -1], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, content)
    listoffsetarray = ak.contents.ListOffsetArray(offsets, indexedarray)

    assert to_list(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, None]]
    assert to_list(listoffsetarray[::-1]) == [[3.3, None], [], [2.2, 2.2, 0.0]]
    assert to_list(listoffsetarray[[2, 0]]) == [[3.3, None], [2.2, 2.2, 0.0]]
    assert to_list(listoffsetarray[[2, 0], 1]) == [None, 2.2]  # invokes carry
    assert to_list(listoffsetarray[2:, 1]) == [None]  # invokes carry


def test_others():
    content = ak.contents.NumpyArray(
        np.array(
            [[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]
        )
    )
    index = ak.index.Index64(np.array([4, 0, 3, 1, 3], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)

    assert indexedarray[3, 0] == 0.1
    assert indexedarray[3, 1] == 1.0
    assert to_list(indexedarray[3, ::-1]) == [1.0, 0.1]
    assert indexedarray.to_typetracer()[3, ::-1].form == indexedarray[3, ::-1].form
    assert to_list(indexedarray[3, [1, 1, 0]]) == [1.0, 1.0, 0.1]
    assert (
        indexedarray.to_typetracer()[3, [1, 1, 0]].form
        == indexedarray[3, [1, 1, 0]].form
    )
    assert to_list(indexedarray[3:, 0]) == [0.1, 0.3]
    assert indexedarray.to_typetracer()[3:, 0].form == indexedarray[3:, 0].form
    assert to_list(indexedarray[3:, 1]) == [1.0, 3.0]
    assert indexedarray.to_typetracer()[3:, 1].form == indexedarray[3:, 1].form
    assert to_list(indexedarray[3:, ::-1]) == [[1.0, 0.1], [3.0, 0.3]]
    assert indexedarray.to_typetracer()[3:, ::-1].form == indexedarray[3:, ::-1].form
    assert to_list(indexedarray[3:, [1, 1, 0]]) == [[1.0, 1.0, 0.1], [3.0, 3.0, 0.3]]
    assert (
        indexedarray.to_typetracer()[3:, [1, 1, 0]].form
        == indexedarray[3:, [1, 1, 0]].form
    )


def test_missing():
    content = ak.contents.NumpyArray(
        np.array(
            [[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]
        )
    )
    index = ak.index.Index64(np.array([4, 0, 3, -1, 3], dtype=np.int64))
    indexedarray = ak.contents.IndexedOptionArray(index, content)

    assert to_list(indexedarray[3:, 0]) == [None, 0.3]
    assert indexedarray.to_typetracer()[3:, 0].form == indexedarray[3:, 0].form
    assert to_list(indexedarray[3:, 1]) == [None, 3.0]
    assert indexedarray.to_typetracer()[3:, 1].form == indexedarray[3:, 1].form
    assert to_list(indexedarray[3:, ::-1]) == [None, [3.0, 0.3]]
    assert indexedarray.to_typetracer()[3:, ::-1].form == indexedarray[3:, ::-1].form
    assert to_list(indexedarray[3:, [1, 1, 0]]) == [None, [3.0, 3.0, 0.3]]
    assert (
        indexedarray.to_typetracer()[3:, [1, 1, 0]].form
        == indexedarray[3:, [1, 1, 0]].form
    )


def test_builder():
    assert to_list(
        ak.highlevel.Array([1.1, 2.2, 3.3, None, 4.4], check_valid=True)
    ) == [
        1.1,
        2.2,
        3.3,
        None,
        4.4,
    ]
    assert to_list(
        ak.highlevel.Array([None, 2.2, 3.3, None, 4.4], check_valid=True)
    ) == [
        None,
        2.2,
        3.3,
        None,
        4.4,
    ]

    assert to_list(
        ak.highlevel.Array([[1.1, 2.2, 3.3], [], [None, 4.4]], check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array([[1.1, 2.2, 3.3], [], None, [None, 4.4]], check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert to_list(
        ak.highlevel.Array([[1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) == [[1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array([[1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) != [[1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert to_list(
        ak.highlevel.Array([[None, 1.1, 2.2, 3.3], [], [None, 4.4]], check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]], check_valid=True
        )
    ) == [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True
        )
    ) == [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True
        )
    ) != [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert to_list(
        ak.highlevel.Array([None, [1.1, 2.2, 3.3], [], [None, 4.4]], check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]], check_valid=True
        )
    ) == [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True
        )
    ) == [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True
        )
    ) != [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert to_list(
        ak.highlevel.Array([None, None, None, None, None], check_valid=True)
    ) == [
        None,
        None,
        None,
        None,
        None,
    ]
    assert to_list(
        ak.highlevel.Array([[None, None, None], [], [None, None]], check_valid=True)
    ) == [[None, None, None], [], [None, None]]


def test_json():
    assert to_list(
        ak.highlevel.Array("[1.1, 2.2, 3.3, null, 4.4]", check_valid=True)
    ) == [
        1.1,
        2.2,
        3.3,
        None,
        4.4,
    ]
    assert to_list(
        ak.highlevel.Array("[null, 2.2, 3.3, null, 4.4]", check_valid=True)
    ) == [
        None,
        2.2,
        3.3,
        None,
        4.4,
    ]

    assert to_list(
        ak.highlevel.Array("[[1.1, 2.2, 3.3], [], [null, 4.4]]", check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array("[[1.1, 2.2, 3.3], [], null, [null, 4.4]]", check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert to_list(
        ak.highlevel.Array("[[1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) == [[1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array("[[1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) != [[1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert to_list(
        ak.highlevel.Array("[[null, 1.1, 2.2, 3.3], [], [null, 4.4]]", check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            "[[null, 1.1, 2.2, 3.3], [], null, [null, 4.4]]", check_valid=True
        )
    ) == [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            "[[null, 1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True
        )
    ) == [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            "[[null, 1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True
        )
    ) != [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert to_list(
        ak.highlevel.Array("[null, [1.1, 2.2, 3.3], [], [null, 4.4]]", check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            "[null, [1.1, 2.2, 3.3], [], null, [null, 4.4]]", check_valid=True
        )
    ) == [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            "[null, [1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True
        )
    ) == [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert to_list(
        ak.highlevel.Array(
            "[null, [1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True
        )
    ) != [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert to_list(
        ak.highlevel.Array("[null, null, null, null, null]", check_valid=True)
    ) == [
        None,
        None,
        None,
        None,
        None,
    ]
    assert to_list(
        ak.highlevel.Array("[[null, null, null], [], [null, null]]", check_valid=True)
    ) == [[None, None, None], [], [None, None]]
