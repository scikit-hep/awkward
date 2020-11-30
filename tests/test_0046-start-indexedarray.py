# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward1 as ak  # noqa: F401


def test_basic():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int32)
    index = ak.layout.Index32(ind)
    array = ak.layout.IndexedArray32(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.uint32)
    index = ak.layout.IndexU32(ind)
    array = ak.layout.IndexedArrayU32(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int64)
    index = ak.layout.Index64(ind)
    array = ak.layout.IndexedArray64(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int32)
    index = ak.layout.Index32(ind)
    array = ak.layout.IndexedOptionArray32(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]

    ind = np.array([2, 2, 0, 3, 4], dtype=np.int64)
    index = ak.layout.Index64(ind)
    array = ak.layout.IndexedOptionArray64(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 3.3, 4.4]
    ind[3] = 1
    assert ak.to_list(array) == [2.2, 2.2, 0.0, 1.1, 4.4]


def test_type():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.layout.Index32(np.array([2, 2, 0, 3, 4], dtype=np.int32))
    array = ak.layout.IndexedArray32(index, content)
    assert ak.type(array) == ak.types.PrimitiveType("float64")
    array = ak.layout.IndexedOptionArray32(index, content)
    assert ak.type(array) == ak.types.OptionType(ak.types.PrimitiveType("float64"))


def test_null():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.layout.Index64(np.array([2, 2, 0, -1, 4], dtype=np.int64))
    array = ak.layout.IndexedOptionArray64(index, content)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, None, 4.4]


def test_carry():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.layout.Index64(np.array([2, 2, 0, 3, 4], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, indexedarray)
    assert ak.to_list(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, 4.4]]
    assert ak.to_list(listoffsetarray[::-1]) == [[3.3, 4.4], [], [2.2, 2.2, 0.0]]
    assert ak.to_list(listoffsetarray[[2, 0]]) == [[3.3, 4.4], [2.2, 2.2, 0.0]]
    assert ak.to_list(listoffsetarray[[2, 0], 1]) == [4.4, 2.2]  # invokes carry
    assert ak.to_list(listoffsetarray[2:, 1]) == [4.4]  # invokes carry

    index = ak.layout.Index64(np.array([2, 2, 0, 3, -1], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    listoffsetarray = ak.layout.ListOffsetArray64(offsets, indexedarray)
    assert ak.to_list(listoffsetarray) == [[2.2, 2.2, 0.0], [], [3.3, None]]
    assert ak.to_list(listoffsetarray[::-1]) == [[3.3, None], [], [2.2, 2.2, 0.0]]
    assert ak.to_list(listoffsetarray[[2, 0]]) == [[3.3, None], [2.2, 2.2, 0.0]]
    assert ak.to_list(listoffsetarray[[2, 0], 1]) == [None, 2.2]  # invokes carry
    assert ak.to_list(listoffsetarray[2:, 1]) == [None]  # invokes carry


def test_others():
    content = ak.layout.NumpyArray(
        np.array(
            [[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]
        )
    )
    index = ak.layout.Index64(np.array([4, 0, 3, 1, 3], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)
    assert indexedarray[3, 0] == 0.1
    assert indexedarray[3, 1] == 1.0
    assert ak.to_list(indexedarray[3, ::-1]) == [1.0, 0.1]
    assert ak.to_list(indexedarray[3, [1, 1, 0]]) == [1.0, 1.0, 0.1]
    assert ak.to_list(indexedarray[3:, 0]) == [0.1, 0.3]
    assert ak.to_list(indexedarray[3:, 1]) == [1.0, 3.0]
    assert ak.to_list(indexedarray[3:, ::-1]) == [[1.0, 0.1], [3.0, 0.3]]
    assert ak.to_list(indexedarray[3:, [1, 1, 0]]) == [[1.0, 1.0, 0.1], [3.0, 3.0, 0.3]]


def test_missing():
    content = ak.layout.NumpyArray(
        np.array(
            [[0.0, 0.0], [0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]]
        )
    )
    index = ak.layout.Index64(np.array([4, 0, 3, -1, 3], dtype=np.int64))
    indexedarray = ak.layout.IndexedOptionArray64(index, content)
    assert ak.to_list(indexedarray[3:, 0]) == [None, 0.3]
    assert ak.to_list(indexedarray[3:, 1]) == [None, 3.0]
    assert ak.to_list(indexedarray[3:, ::-1]) == [None, [3.0, 0.3]]
    assert ak.to_list(indexedarray[3:, [1, 1, 0]]) == [None, [3.0, 3.0, 0.3]]


def test_highlevel():
    content = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    index = ak.layout.Index64(np.array([2, 2, 0, -1, 4], dtype=np.int64))
    array = ak.Array(ak.layout.IndexedOptionArray64(index, content), check_valid=True)
    assert ak.to_list(array) == [2.2, 2.2, 0.0, None, 4.4]
    assert str(array) == "[2.2, 2.2, 0, None, 4.4]"
    assert repr(array) == "<Array [2.2, 2.2, 0, None, 4.4] type='5 * ?float64'>"


def test_builder():
    assert ak.to_list(ak.Array([1.1, 2.2, 3.3, None, 4.4], check_valid=True)) == [
        1.1,
        2.2,
        3.3,
        None,
        4.4,
    ]
    assert ak.to_list(ak.Array([None, 2.2, 3.3, None, 4.4], check_valid=True)) == [
        None,
        2.2,
        3.3,
        None,
        4.4,
    ]

    assert ak.to_list(
        ak.Array([[1.1, 2.2, 3.3], [], [None, 4.4]], check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], [None, 4.4]]
    assert ak.to_list(
        ak.Array([[1.1, 2.2, 3.3], [], None, [None, 4.4]], check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert ak.to_list(
        ak.Array([[1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) == [[1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert ak.to_list(
        ak.Array([[1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) != [[1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert ak.to_list(
        ak.Array([[None, 1.1, 2.2, 3.3], [], [None, 4.4]], check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], [], [None, 4.4]]
    assert ak.to_list(
        ak.Array([[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]], check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert ak.to_list(
        ak.Array([[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert ak.to_list(
        ak.Array([[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) != [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert ak.to_list(
        ak.Array([None, [1.1, 2.2, 3.3], [], [None, 4.4]], check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], [], [None, 4.4]]
    assert ak.to_list(
        ak.Array([None, [1.1, 2.2, 3.3], [], None, [None, 4.4]], check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert ak.to_list(
        ak.Array([None, [1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert ak.to_list(
        ak.Array([None, [1.1, 2.2, 3.3], None, [], [None, 4.4]], check_valid=True)
    ) != [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert ak.to_list(ak.Array([None, None, None, None, None], check_valid=True)) == [
        None,
        None,
        None,
        None,
        None,
    ]
    assert ak.to_list(
        ak.Array([[None, None, None], [], [None, None]], check_valid=True)
    ) == [[None, None, None], [], [None, None]]


def test_json():
    assert ak.to_list(ak.Array("[1.1, 2.2, 3.3, null, 4.4]", check_valid=True)) == [
        1.1,
        2.2,
        3.3,
        None,
        4.4,
    ]
    assert ak.to_list(ak.Array("[null, 2.2, 3.3, null, 4.4]", check_valid=True)) == [
        None,
        2.2,
        3.3,
        None,
        4.4,
    ]

    assert ak.to_list(
        ak.Array("[[1.1, 2.2, 3.3], [], [null, 4.4]]", check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], [None, 4.4]]
    assert ak.to_list(
        ak.Array("[[1.1, 2.2, 3.3], [], null, [null, 4.4]]", check_valid=True)
    ) == [[1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert ak.to_list(
        ak.Array("[[1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) == [[1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert ak.to_list(
        ak.Array("[[1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) != [[1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert ak.to_list(
        ak.Array("[[null, 1.1, 2.2, 3.3], [], [null, 4.4]]", check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], [], [None, 4.4]]
    assert ak.to_list(
        ak.Array("[[null, 1.1, 2.2, 3.3], [], null, [null, 4.4]]", check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert ak.to_list(
        ak.Array("[[null, 1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) == [[None, 1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert ak.to_list(
        ak.Array("[[null, 1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) != [[None, 1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert ak.to_list(
        ak.Array("[null, [1.1, 2.2, 3.3], [], [null, 4.4]]", check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], [], [None, 4.4]]
    assert ak.to_list(
        ak.Array("[null, [1.1, 2.2, 3.3], [], null, [null, 4.4]]", check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]
    assert ak.to_list(
        ak.Array("[null, [1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) == [None, [1.1, 2.2, 3.3], None, [], [None, 4.4]]
    assert ak.to_list(
        ak.Array("[null, [1.1, 2.2, 3.3], null, [], [null, 4.4]]", check_valid=True)
    ) != [None, [1.1, 2.2, 3.3], [], None, [None, 4.4]]

    assert ak.to_list(ak.Array("[null, null, null, null, null]", check_valid=True)) == [
        None,
        None,
        None,
        None,
        None,
    ]
    assert ak.to_list(
        ak.Array("[[null, null, null], [], [null, null]]", check_valid=True)
    ) == [[None, None, None], [], [None, None]]
