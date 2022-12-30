# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("awkward._connect.pyarrow")

to_list = ak.operations.to_list


def test_toarrow_BitMaskedArray():
    content = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    bitmask = ak.index.IndexU8(np.array([40, 34], dtype=np.uint8))
    array = ak.contents.BitMaskedArray(bitmask, content, False, 9, False)
    assert array.to_arrow().to_pylist() == to_list(array)


def test_toarrow_ByteMaskedArray_1():
    content = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    bytemask = ak.index.Index8(np.array([False, True, False], dtype=np.bool_))
    array = ak.contents.ByteMaskedArray(bytemask, content, True)
    assert array.to_arrow().to_pylist() == to_list(array)


def test_toarrow_NumpyArray_1():
    array = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5]))
    assert isinstance(array.to_arrow(), pyarrow.lib.Array)
    assert array.to_arrow().to_pylist() == [0.0, 1.1, 2.2, 3.3, 4.4, 5.5]


def test_toarrow_NumpyArray_2():
    array = ak.contents.NumpyArray(np.array([[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]))
    assert isinstance(array.to_arrow(), pyarrow.lib.Array)
    assert array.to_arrow().to_pylist() == [[0.0, 1.1], [2.2, 3.3], [4.4, 5.5]]


def test_toarrow_EmptyArray():
    array = ak.contents.EmptyArray()
    assert isinstance(array.to_arrow(), pyarrow.lib.Array)
    assert array.to_arrow().to_pylist() == []


def test_toarrow_ListOffsetArray64():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.contents.ListOffsetArray(offsets, content)
    assert isinstance(array.to_arrow().storage, pyarrow.LargeListArray)
    assert array.to_arrow().to_pylist() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert array[1:].to_arrow().to_pylist() == [
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert array[2:].to_arrow().to_pylist() == [
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]


def test_toarrow_ListOffsetArrayU32():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.IndexU32(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.contents.ListOffsetArray(offsets, content)
    assert isinstance(array.to_arrow().storage, pyarrow.ListArray)
    assert array.to_arrow().to_pylist() == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert array[1:].to_arrow().to_pylist() == [
        [],
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]
    assert array[2:].to_arrow().to_pylist() == [
        [4.4, 5.5],
        [6.6],
        [7.7, 8.8, 9.9],
    ]


def test_toarrow_ListArray_RegularArray():
    content = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    array = ak.contents.ListOffsetArray(offsets, content)
    assert array.to_arrow().to_pylist() == [
        ["one", "two", "three"],
        [],
        ["four", "five"],
        ["six"],
        ["seven", "eight", "nine"],
    ]

    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    starts = ak.index.Index64(np.array([0, 1], dtype=np.int64))
    stops = ak.index.Index64(np.array([2, 3], dtype=np.int64))
    listarray = ak.contents.ListArray(starts, stops, regulararray)

    assert isinstance(listarray.to_arrow().storage, pyarrow.LargeListArray)
    assert listarray.to_arrow().to_pylist() == [
        [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]],
        [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]],
    ]
    assert listarray[1:].to_arrow().to_pylist() == [
        [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]],
    ]

    assert isinstance(regulararray.to_arrow().storage, pyarrow.FixedSizeListArray)
    assert regulararray.to_arrow().to_pylist() == [
        [[0.0, 1.1, 2.2], []],
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]
    assert regulararray[1:].to_arrow().to_pylist() == [
        [[3.3, 4.4], [5.5]],
        [[6.6, 7.7, 8.8, 9.9], []],
    ]


def test_toarrow_RecordArray():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))

    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "two", "2", "wonky"],
    )

    assert isinstance(recordarray.to_arrow().storage, pyarrow.StructArray)
    assert recordarray.to_arrow().to_pylist() == [
        {"one": 1, "two": [0.0, 1.1, 2.2], "2": 1.1, "wonky": 1},
        {"one": 2, "two": [], "2": 2.2, "wonky": 2},
        {"one": 3, "two": [3.3, 4.4], "2": 3.3, "wonky": 3},
        {"one": 4, "two": [5.5], "2": 4.4, "wonky": 4},
        {"one": 5, "two": [6.6, 7.7, 8.8, 9.9], "2": 5.5, "wonky": 5},
    ]


def test_toarrow_UnionArray():
    content0 = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    unionarray = ak.contents.UnionArray(tags, index, [content0, content1])

    assert isinstance(unionarray.to_arrow().storage, pyarrow.UnionArray)
    assert unionarray.to_arrow().to_pylist() == [
        1,
        2,
        [1.1, 2.2, 3.3],
        [],
        3,
        [4.4, 5.5],
        5,
        4,
    ]


def test_toarrow_IndexedArray():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index = ak.index.Index32(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)

    assert isinstance(indexedarray.to_arrow().storage, pyarrow.lib.DoubleArray)
    assert indexedarray.to_arrow().to_pylist() == [
        0.0,
        2.2,
        4.4,
        6.6,
        8.8,
        9.9,
        7.7,
        5.5,
    ]


def test_toarrow_IndexedOptionArray_2():
    array = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5, None]).layout

    assert array.to_arrow().to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5, None]
    assert array[:-1].to_arrow().to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert array[:1].to_arrow().to_pylist() == [1.1]
    assert array[:0].to_arrow().to_pylist() == []

    content = ak.contents.NumpyArray(np.array([], dtype=np.float64))
    index = ak.index.Index32(np.array([-1, -1, -1, -1], dtype=np.int32))
    indexedoptionarray = ak.contents.IndexedOptionArray(index, content)
    assert indexedoptionarray.to_arrow().to_pylist() == [None, None, None, None]


def test_toarrow_ByteMaskedArray_2():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, True, False, False, False], dtype=np.int8)),
        listoffsetarray,
        True,
    )

    assert bytemaskedarray.to_arrow().to_pylist() == [
        [0.0, 1.1, 2.2],
        [],
        None,
        None,
        None,
    ]


def test_toarrow_ByteMaskedArray_3():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    starts = ak.index.Index64(np.array([0, 1]))
    stops = ak.index.Index64(np.array([2, 3]))
    listarray = ak.contents.ListArray(starts, stops, regulararray)

    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False], dtype=np.int8)), listarray, True
    )
    assert bytemaskedarray.to_arrow().to_pylist() == to_list(bytemaskedarray)


def test_toarrow_ByteMaskedArray_4():
    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "two", "2", "wonky"],
    )

    bytemaskedarray = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([True, False], dtype=np.int8)), recordarray, True
    )
    assert bytemaskedarray.to_arrow().to_pylist() == to_list(bytemaskedarray)


def test_toarrow_ByteMaskedArray_5():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    index = ak.index.Index32(np.array([0, 2, 4, 6, 8, 9, 7, 5], dtype=np.int64))
    indexedarray = ak.contents.IndexedArray(index, content)

    bytemaskedarray = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(np.array([True, False, False], dtype=np.int8)),
        indexedarray,
        True,
    )
    assert bytemaskedarray.to_arrow().to_pylist() == to_list(bytemaskedarray)


def test_toarrow_ByteMaskedArray_broken_unions_1():
    content0 = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    ).layout
    content1 = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 1, 0, 2, 2, 4, 3, 3, 4], dtype=np.int32))
    unionarray = ak.contents.UnionArray(tags, index, [content0, content1])

    bytemaskedarray = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(
            # tags          1,     1,     0,    0,     1,    0,    1,     1,    0,     0
            # index         0,     1,     1,    0,     2,    2,    4,     3,    3,     4
            np.array(
                [True, False, False, True, False, True, True, False, False, True],
                dtype=np.int8,
            )
        ),
        unionarray,
        valid_when=True,
    )

    assert bytemaskedarray.to_arrow().to_pylist() == to_list(bytemaskedarray)


def test_toarrow_ByteMaskedArray_broken_unions_2():
    content0 = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    ).layout
    content1 = ak.contents.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4]))
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int8))
    index = ak.index.Index32(
        np.array([0, 1, 1, 0, 2, 2, 4, 3, 3, 4, 3], dtype=np.int32)
    )
    unionarray = ak.contents.UnionArray(tags, index, [content0, content1])

    bytemaskedarray = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(
            # tags          1,     1,     0,    0,     1,    0,    1,     1,    0,     0,    0
            # index         0,     1,     1,    0,     2,    2,    4,     3,    3,     4,    3
            np.array(
                [True, False, False, True, False, True, True, False, False, True, True],
                dtype=np.int8,
            )
        ),
        unionarray,
        valid_when=True,
    )
    assert bytemaskedarray.to_arrow().to_pylist() == to_list(bytemaskedarray)


def test_toarrow_IndexedOptionArray():
    ioa = ak.contents.IndexedOptionArray(
        ak.index.Index32([-30, 19, 6, 7, -3, 21, 13, 22, 17, 9, -12, 16]),
        ak.contents.NumpyArray(
            np.array(
                [
                    5.2,
                    1.7,
                    6.7,
                    -0.4,
                    4.0,
                    7.8,
                    3.8,
                    6.8,
                    4.2,
                    0.3,
                    4.6,
                    6.2,
                    6.9,
                    -0.7,
                    3.9,
                    1.6,
                    8.7,
                    -0.7,
                    3.2,
                    4.3,
                    4.0,
                    5.8,
                    4.2,
                    7.0,
                    5.6,
                    3.8,
                ]
            )
        ),
    )
    assert ioa.to_arrow().to_pylist() == to_list(ioa)


def test_fromarrow_NumpyArray_1():
    boolarray = ak.contents.NumpyArray(
        np.array([True, True, True, False, False, True, False, True, False, True])
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(boolarray.to_arrow())) == to_list(
        boolarray
    )


def test_fromarrow_NumpyArray_2():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(content.to_arrow())) == to_list(
        content
    )


def test_fromarrow_ListOffsetArray():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(listoffsetarray.to_arrow())
    ) == to_list(listoffsetarray)


def test_fromarrow_RegularArray():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(regulararray.to_arrow())
    ) == to_list(regulararray)


def test_fromarrow_RecordArray():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10], dtype=np.int64))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "chonks", "2", "wonky"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(recordarray.to_arrow())) == to_list(
        recordarray
    )


def test_fromarrow_UnionArray():
    content0 = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.contents.UnionArray(tags, index, [content0, content])
    assert to_list(ak._connect.pyarrow.handle_arrow(array.to_arrow())) == to_list(array)


def test_chunkedarray():
    a = pyarrow.chunked_array(
        [
            pyarrow.array([1.1, 2.2, 3.3]),
            pyarrow.array([], pyarrow.float64()),
            pyarrow.array([4.4, 5.5]),
            pyarrow.array([6.6]),
            pyarrow.array([], pyarrow.float64()),
            pyarrow.array([], pyarrow.float64()),
            pyarrow.array([7.7, 8.8, 9.9]),
        ]
    )
    assert a.to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]


def test_recordbatch():
    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            pyarrow.array([[1, 2, 3], [], [], [4, 5], [6]]),
        ],
        ["a", "b"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"a": 1.1, "b": [1, 2, 3]},
        {"a": 2.2, "b": []},
        {"a": 3.3, "b": []},
        {"a": 4.4, "b": [4, 5]},
        {"a": 5.5, "b": [6]},
    ]

    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
            pyarrow.array([[1, None, 3], [], [], [4, 5], [6]]),
        ],
        ["a", "b"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"a": 1.1, "b": [1, None, 3]},
        {"a": 2.2, "b": []},
        {"a": 3.3, "b": []},
        {"a": None, "b": [4, 5]},
        {"a": 5.5, "b": [6]},
    ]

    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
            pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
            pyarrow.array(
                [
                    {"x": 1, "y": 1.1},
                    {"x": 2, "y": 2.2},
                    {"x": 3, "y": 3.3},
                    {"x": 4, "y": None},
                    {"x": 5, "y": 5.5},
                ]
            ),
            pyarrow.array(
                [
                    {"x": 1, "y": 1.1},
                    None,
                    None,
                    {"x": 4, "y": None},
                    {"x": 5, "y": 5.5},
                ]
            ),
            pyarrow.array(
                [
                    [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                    [],
                    [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
                    [None],
                    [{"x": 6, "y": 6.6}],
                ]
            ),
        ],
        ["a", "b", "c", "d", "e"],
    )

    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {
            "a": 1.1,
            "b": [1, 2, 3],
            "c": {"x": 1, "y": 1.1},
            "d": {"x": 1, "y": 1.1},
            "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        },
        {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []},
        {
            "a": 3.3,
            "b": [4, 5],
            "c": {"x": 3, "y": 3.3},
            "d": None,
            "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
        },
        {
            "a": None,
            "b": [None],
            "c": {"x": 4, "y": None},
            "d": {"x": 4, "y": None},
            "e": [None],
        },
        {
            "a": 5.5,
            "b": [6],
            "c": {"x": 5, "y": 5.5},
            "d": {"x": 5, "y": 5.5},
            "e": [{"x": 6, "y": 6.6}],
        },
    ]


### All of the following tests were copied (translated) over from Awkward 0.


def test_arrow_toarrow_string():
    a = ak.operations.from_iter(["one", "two", "three"]).layout
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == to_list(a)
    a = ak.operations.from_iter([["one", "two", "three"], [], ["four", "five"]]).layout
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == to_list(a)
    if hasattr(pyarrow.BinaryArray, "from_buffers"):
        a = ak.operations.from_iter([b"one", b"two", b"three"]).layout
        assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
            b"one",
            b"two",
            b"three",
        ]
        a = ak.operations.from_iter(
            [[b"one", b"two", b"three"], [], [b"four", b"five"]]
        ).layout
        assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
            [b"one", b"two", b"three"],
            [],
            [b"four", b"five"],
        ]
    else:
        a = ak.operations.from_iter([b"one", b"two", b"three"]).layout
        assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
            "one",
            "two",
            "three",
        ]
        a = ak.operations.from_iter(
            [[b"one", b"two", b"three"], [], [b"four", b"five"]]
        ).layout
        assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
            ["one", "two", "three"],
            [],
            ["four", "five"],
        ]


def test_arrow_array():
    a = pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]


def test_arrow_boolean():
    a = pyarrow.array([True, True, False, False, True])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        True,
        True,
        False,
        False,
        True,
    ]


def test_arrow_array_null():
    a = pyarrow.array([1.1, 2.2, 3.3, None, 4.4, 5.5])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        1.1,
        2.2,
        3.3,
        None,
        4.4,
        5.5,
    ]


def test_arrow_nested_array():
    a = pyarrow.array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]


def test_arrow_nested_nested_array():
    a = pyarrow.array([[[1.1, 2.2], [3.3], []], [], [[4.4, 5.5]]])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [[1.1, 2.2], [3.3], []],
        [],
        [[4.4, 5.5]],
    ]


def test_arrow_nested_array_null():
    a = pyarrow.array([[1.1, 2.2, None], [], [4.4, 5.5]])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [1.1, 2.2, None],
        [],
        [4.4, 5.5],
    ]


def test_arrow_null_nested_array_null():
    a = pyarrow.array([[1.1, 2.2, None], [], None, [4.4, 5.5]])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [1.1, 2.2, None],
        [],
        None,
        [4.4, 5.5],
    ]


def test_arrow_chunked_array():
    a = pyarrow.chunked_array(
        [
            pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            pyarrow.array([], pyarrow.float64()),
            pyarrow.array([6.6, 7.7, 8.8]),
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
    ]


def test_arrow_struct():
    a = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]


def test_arrow_struct_null():
    a = pyarrow.array([{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": None},
        {"x": 3, "y": 3.3},
    ]


def test_arrow_null_struct():
    a = pyarrow.array(
        [{"x": 1, "y": 1.1}, None, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"x": 1, "y": 1.1},
        None,
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]


def test_arrow_null_struct_null():
    a = pyarrow.array(
        [{"x": 1, "y": 1.1}, None, {"x": 2, "y": None}, {"x": 3, "y": 3.3}]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"x": 1, "y": 1.1},
        None,
        {"x": 2, "y": None},
        {"x": 3, "y": 3.3},
    ]


def test_arrow_chunked_struct():
    t = pyarrow.struct({"x": pyarrow.int64(), "y": pyarrow.float64()})
    a = pyarrow.chunked_array(
        [
            pyarrow.array(
                [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], t
            ),
            pyarrow.array([], t),
            pyarrow.array([{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}], t),
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
        {"x": 4, "y": 4.4},
        {"x": 5, "y": 5.5},
    ]


def test_arrow_nested_struct():
    a = pyarrow.array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
            [],
            [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ]


def test_arrow_nested_struct_null():
    a = pyarrow.array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}],
            [],
            [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}],
        [],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ]


def test_arrow_null_nested_struct():
    a = pyarrow.array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
            None,
            [],
            [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        None,
        [],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ]


def test_arrow_null_nested_struct_null():
    a = pyarrow.array(
        [
            [{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}],
            None,
            [],
            [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [{"x": 1, "y": 1.1}, {"x": 2, "y": None}, {"x": 3, "y": 3.3}],
        None,
        [],
        [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}],
    ]


def test_arrow_struct_nested():
    a = pyarrow.array(
        [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"x": [], "y": 1.1},
        {"x": [2], "y": 2.2},
        {"x": [3, 3], "y": 3.3},
    ]


def test_arrow_struct_nested_null():
    a = pyarrow.array(
        [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {"x": [], "y": 1.1},
        {"x": [2], "y": 2.2},
        {"x": [None, 3], "y": 3.3},
    ]


def test_arrow_nested_struct_nested():
    a = pyarrow.array(
        [
            [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}],
            [],
            [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}],
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [3, 3], "y": 3.3}],
        [],
        [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}],
    ]


def test_arrow_null_nested_struct_nested_null():
    a = pyarrow.array(
        [
            [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}],
            None,
            [],
            [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}],
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [{"x": [], "y": 1.1}, {"x": [2], "y": 2.2}, {"x": [None, 3], "y": 3.3}],
        None,
        [],
        [{"x": [4, 4, 4], "y": 4.4}, {"x": [5, 5, 5, 5], "y": 5.5}],
    ]


def test_arrow_strings():
    a = pyarrow.array(["one", "two", "three", "fo\u2014ur", "five"])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        "one",
        "two",
        "three",
        "fo\u2014ur",
        "five",
    ]


def test_arrow_strings_null():
    a = pyarrow.array(["one", "two", None, "fo\u2014ur", "five"])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        "one",
        "two",
        None,
        "fo\u2014ur",
        "five",
    ]


def test_arrow_binary():
    a = pyarrow.array([b"one", b"two", b"three", b"four", b"five"])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        b"one",
        b"two",
        b"three",
        b"four",
        b"five",
    ]


def test_arrow_binary_null():
    a = pyarrow.array([b"one", b"two", None, b"four", b"five"])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        b"one",
        b"two",
        None,
        b"four",
        b"five",
    ]


def test_arrow_chunked_strings():
    a = pyarrow.chunked_array(
        [
            pyarrow.array(["one", "two", "three", "four", "five"]),
            pyarrow.array(["six", "seven", "eight"]),
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
    ]


def test_arrow_nested_strings():
    a = pyarrow.array([["one", "two", "three"], [], ["four", "five"]])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        ["one", "two", "three"],
        [],
        ["four", "five"],
    ]


def test_arrow_nested_strings_null():
    a = pyarrow.array([["one", "two", None], [], ["four", "five"]])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        ["one", "two", None],
        [],
        ["four", "five"],
    ]


def test_arrow_null_nested_strings_null():
    a = pyarrow.array([["one", "two", None], [], None, ["four", "five"]])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        ["one", "two", None],
        [],
        None,
        ["four", "five"],
    ]


def test_arrow_union_sparse():
    a = pyarrow.UnionArray.from_sparse(
        pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()),
        [
            pyarrow.array([0.0, 1.1, 2.2, 3.3, 4.4]),
            pyarrow.array([True, True, False, True, False]),
        ],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        0.0,
        True,
        2.2,
        3.3,
        False,
    ]


def test_arrow_union_sparse_null():
    a = pyarrow.UnionArray.from_sparse(
        pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()),
        [
            pyarrow.array([0.0, 1.1, None, 3.3, 4.4]),
            pyarrow.array([True, True, False, True, False]),
        ],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        0.0,
        True,
        None,
        3.3,
        False,
    ]


def test_arrow_union_sparse_null_null():
    a = pyarrow.UnionArray.from_sparse(
        pyarrow.array([0, 1, 0, 0, 1], type=pyarrow.int8()),
        [
            pyarrow.array([0.0, 1.1, None, 3.3, 4.4]),
            pyarrow.array([True, None, False, True, False]),
        ],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        0.0,
        None,
        None,
        3.3,
        False,
    ]


def test_arrow_union_dense():
    a = pyarrow.UnionArray.from_dense(
        pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()),
        pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()),
        [pyarrow.array([0.0, 1.1, 2.2, 3.3]), pyarrow.array([True, True, False])],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        0.0,
        True,
        1.1,
        2.2,
        3.3,
        True,
        False,
    ]


def test_arrow_union_dense_null():
    a = pyarrow.UnionArray.from_dense(
        pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()),
        pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()),
        [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, True, False])],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        0.0,
        True,
        1.1,
        None,
        3.3,
        True,
        False,
    ]


def test_arrow_union_dense_null_null():
    a = pyarrow.UnionArray.from_dense(
        pyarrow.array([0, 1, 0, 0, 0, 1, 1], type=pyarrow.int8()),
        pyarrow.array([0, 0, 1, 2, 3, 1, 2], type=pyarrow.int32()),
        [pyarrow.array([0.0, 1.1, None, 3.3]), pyarrow.array([True, None, False])],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        0.0,
        True,
        1.1,
        None,
        3.3,
        None,
        False,
    ]


def test_arrow_dictarray():
    a = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]),
        pyarrow.array(["one", "two", "three"]),
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        "one",
        "one",
        "three",
        "three",
        "two",
        "one",
        "three",
        "two",
        "two",
    ]


def test_arrow_dictarray_null():
    a = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array([0, 0, 2, None, 1, None, 2, 1, 1]),
        pyarrow.array(["one", "two", "three"]),
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        "one",
        "one",
        "three",
        None,
        "two",
        None,
        "three",
        "two",
        "two",
    ]


def test_arrow_null_dictarray():
    a = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]),
        pyarrow.array(["one", None, "three"]),
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        "one",
        "one",
        "three",
        "three",
        None,
        "one",
        "three",
        None,
        None,
    ]


def test_arrow_batch():
    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
            pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
            pyarrow.array(
                [
                    {"x": 1, "y": 1.1},
                    {"x": 2, "y": 2.2},
                    {"x": 3, "y": 3.3},
                    {"x": 4, "y": None},
                    {"x": 5, "y": 5.5},
                ]
            ),
            pyarrow.array(
                [
                    {"x": 1, "y": 1.1},
                    None,
                    None,
                    {"x": 4, "y": None},
                    {"x": 5, "y": 5.5},
                ]
            ),
            pyarrow.array(
                [
                    [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
                    [],
                    [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
                    [None],
                    [{"x": 6, "y": 6.6}],
                ]
            ),
        ],
        ["a", "b", "c", "d", "e"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {
            "a": 1.1,
            "b": [1, 2, 3],
            "c": {"x": 1, "y": 1.1},
            "d": {"x": 1, "y": 1.1},
            "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        },
        {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []},
        {
            "a": 3.3,
            "b": [4, 5],
            "c": {"x": 3, "y": 3.3},
            "d": None,
            "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
        },
        {
            "a": None,
            "b": [None],
            "c": {"x": 4, "y": None},
            "d": {"x": 4, "y": None},
            "e": [None],
        },
        {
            "a": 5.5,
            "b": [6],
            "c": {"x": 5, "y": 5.5},
            "d": {"x": 5, "y": 5.5},
            "e": [{"x": 6, "y": 6.6}],
        },
    ]


def test_arrow_table():
    a = pyarrow.Table.from_batches(
        [
            pyarrow.RecordBatch.from_arrays(
                [
                    pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                    pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            {"x": 2, "y": 2.2},
                            {"x": 3, "y": 3.3},
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            None,
                            None,
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            [
                                {"x": 1, "y": 1.1},
                                {"x": 2, "y": 2.2},
                                {"x": 3, "y": 3.3},
                            ],
                            [],
                            [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
                            [None],
                            [{"x": 6, "y": 6.6}],
                        ]
                    ),
                ],
                ["a", "b", "c", "d", "e"],
            ),
            pyarrow.RecordBatch.from_arrays(
                [
                    pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                    pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            {"x": 2, "y": 2.2},
                            {"x": 3, "y": 3.3},
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            None,
                            None,
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            [
                                {"x": 1, "y": 1.1},
                                {"x": 2, "y": 2.2},
                                {"x": 3, "y": 3.3},
                            ],
                            [],
                            [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
                            [None],
                            [{"x": 6, "y": 6.6}],
                        ]
                    ),
                ],
                ["a", "b", "c", "d", "e"],
            ),
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        {
            "a": 1.1,
            "b": [1, 2, 3],
            "c": {"x": 1, "y": 1.1},
            "d": {"x": 1, "y": 1.1},
            "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        },
        {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []},
        {
            "a": 3.3,
            "b": [4, 5],
            "c": {"x": 3, "y": 3.3},
            "d": None,
            "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
        },
        {
            "a": None,
            "b": [None],
            "c": {"x": 4, "y": None},
            "d": {"x": 4, "y": None},
            "e": [None],
        },
        {
            "a": 5.5,
            "b": [6],
            "c": {"x": 5, "y": 5.5},
            "d": {"x": 5, "y": 5.5},
            "e": [{"x": 6, "y": 6.6}],
        },
        {
            "a": 1.1,
            "b": [1, 2, 3],
            "c": {"x": 1, "y": 1.1},
            "d": {"x": 1, "y": 1.1},
            "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        },
        {"a": 2.2, "b": [], "c": {"x": 2, "y": 2.2}, "d": None, "e": []},
        {
            "a": 3.3,
            "b": [4, 5],
            "c": {"x": 3, "y": 3.3},
            "d": None,
            "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
        },
        {
            "a": None,
            "b": [None],
            "c": {"x": 4, "y": None},
            "d": {"x": 4, "y": None},
            "e": [None],
        },
        {
            "a": 5.5,
            "b": [6],
            "c": {"x": 5, "y": 5.5},
            "d": {"x": 5, "y": 5.5},
            "e": [{"x": 6, "y": 6.6}],
        },
    ]


def test_arrow_nonnullable_table():
    x = pyarrow.array([1, 2, 3])
    y = pyarrow.array([1.1, 2.2, 3.3])
    table = pyarrow.Table.from_arrays([x], ["x"])
    if hasattr(pyarrow, "column"):
        table2 = table.add_column(
            1,
            pyarrow.column(
                pyarrow.field("y", y.type, False), np.array([1.1, 2.2, 3.3])
            ),
        )
    else:
        table2 = table.add_column(1, "y", y)
    assert to_list(ak._connect.pyarrow.handle_arrow(table2)) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 3, "y": 3.3},
    ]


def test_arrow_coverage100():
    a = ak.operations.from_iter(
        [True, True, False, False, True, False, True, False]
    ).layout
    assert a.to_arrow().to_pylist() == to_list(a)

    a = ak.contents.ListOffsetArray(
        ak.index.Index32(np.array([0, 5, 10], "i4")),
        ak.contents.NumpyArray(
            np.frombuffer(b"hellothere", "u1"), parameters={"__array__": "byte"}
        ),
        parameters={"__array__": "bytestring"},
    )
    assert a.to_arrow().to_pylist() == [b"hello", b"there"]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True, False, False, True, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10, 15, 20, 25, 30], "i4")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellotherehellotherehellothere", "u1"),
                parameters={"__array__": "byte"},
            ),
            parameters={"__array__": "bytestring"},
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == [
        b"hello",
        None,
        b"hello",
        b"there",
        None,
        None,
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10], "i4")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellothere", "u1"), parameters={"__array__": "byte"}
            ),
            parameters={"__array__": "bytestring"},
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == [b"hello", None]

    a = ak.contents.IndexedOptionArray(
        ak.index.Index32(np.array([-1, 1, -1, 0, 0, -1], "i4")),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10], "i4")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellothere", "u1"), parameters={"__array__": "byte"}
            ),
            parameters={"__array__": "bytestring"},
        ),
    )
    assert a.to_arrow().to_pylist() == [
        None,
        b"there",
        None,
        b"hello",
        b"hello",
        None,
    ]

    a = ak.contents.ListOffsetArray(
        ak.index.Index32(np.array([0, 5, 10], "i4")),
        ak.contents.NumpyArray(
            np.frombuffer(b"hellothere", "u1"), parameters={"__array__": "char"}
        ),
        parameters={"__array__": "string"},
    )
    assert a.to_arrow().to_pylist() == ["hello", "there"]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True, False, False, True, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10, 15, 20, 25, 30], "i4")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellotherehellotherehellothere", "u1"),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == ["hello", None, "hello", "there", None, None]
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
        "hello",
        None,
        "hello",
        "there",
        None,
        None,
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True, False, False, True, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], "i8")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellotherehellotherehellothere", "u1"),
                parameters={"__array__": "char"},
            ),
            parameters={"__array__": "string"},
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == ["hello", None, "hello", "there", None, None]
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
        "hello",
        None,
        "hello",
        "there",
        None,
        None,
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True, False, False, True, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 5, 10, 15, 20, 25, 30], "i8")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellotherehellotherehellothere", "u1"),
                parameters={"__array__": "byte"},
            ),
            parameters={"__array__": "bytestring"},
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == [
        b"hello",
        None,
        b"hello",
        b"there",
        None,
        None,
    ]
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
        b"hello",
        None,
        b"hello",
        b"there",
        None,
        None,
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10], "i4")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellothere", "u1"), parameters={"__array__": "char"}
            ),
            parameters={"__array__": "string"},
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == ["hello", None]

    a = ak.contents.IndexedOptionArray(
        ak.index.Index32(np.array([-1, 1, -1, 0, 0, -1], "i4")),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10], "i4")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellothere", "u1"), parameters={"__array__": "char"}
            ),
            parameters={"__array__": "string"},
        ),
    )
    assert a.to_arrow().to_pylist() == [None, "there", None, "hello", "hello", None]

    a = ak.contents.ListOffsetArray(
        ak.index.Index32(np.array([0, 5, 10], "i4")),
        ak.contents.NumpyArray(np.frombuffer(b"hellothere", "u1")),
    )
    assert a.to_arrow().to_pylist() == [
        [104, 101, 108, 108, 111],
        [116, 104, 101, 114, 101],
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True, False, False, True, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10, 15, 20, 25, 30], "i4")),
            ak.contents.NumpyArray(
                np.frombuffer(b"hellotherehellotherehellothere", "u1")
            ),
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == [
        [104, 101, 108, 108, 111],
        None,
        [104, 101, 108, 108, 111],
        [116, 104, 101, 114, 101],
        None,
        None,
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, True])),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10], "i4")),
            ak.contents.NumpyArray(np.frombuffer(b"hellothere", "u1")),
        ),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == [[104, 101, 108, 108, 111], None]

    a = ak.contents.IndexedOptionArray(
        ak.index.Index32(np.array([-1, 1, -1, 0, 0, -1], "i4")),
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10], "i4")),
            ak.contents.NumpyArray(np.frombuffer(b"hellothere", "u1")),
        ),
    )
    assert a.to_arrow().to_pylist() == [
        None,
        [116, 104, 101, 114, 101],
        None,
        [104, 101, 108, 108, 111],
        [104, 101, 108, 108, 111],
        None,
    ]

    a = ak.contents.IndexedOptionArray(
        ak.index.Index32(np.array([-1, 1, -1, 0, 0, -1], "i4")),
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            3,
            zeros_length=0,
        ),
    )
    assert a.to_arrow().to_pylist() == [
        None,
        [4.4, 5.5, 6.6],
        None,
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        None,
    ]

    a = ak.contents.IndexedOptionArray(
        ak.index.Index32(np.array([-1, 1, -1, 0, 0, -1, 1, -1], "i4")),
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            3,
            zeros_length=0,
        ),
    )
    assert a.to_arrow().to_pylist() == [
        None,
        [4.4, 5.5, 6.6],
        None,
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        None,
        [4.4, 5.5, 6.6],
        None,
    ]

    a = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([-1, 1, -1, 0, 0, -1, 1, -1], "i8")),
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
            3,
            zeros_length=0,
        ),
    )
    assert a.to_arrow().to_pylist() == [
        None,
        [4.4, 5.5, 6.6],
        None,
        [1.1, 2.2, 3.3],
        [1.1, 2.2, 3.3],
        None,
        [4.4, 5.5, 6.6],
        None,
    ]

    a = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(np.array([True, True, True, True, False, False])),
        ak.contents.IndexedOptionArray(
            ak.index.Index32(np.array([-1, 1, -1, 0, 0, -1], "i4")),
            ak.contents.RegularArray(
                ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
                3,
                zeros_length=0,
            ),
        ),
        valid_when=True,
    )
    assert a.to_arrow().to_pylist() == [
        None,
        [4.4, 5.5, 6.6],
        None,
        [1.1, 2.2, 3.3],
        None,
        None,
    ]

    a = ak.contents.UnmaskedArray(
        ak.contents.ListOffsetArray(
            ak.index.Index32(np.array([0, 5, 10], "i4")),
            ak.contents.NumpyArray(np.frombuffer(b"hellothere", "u1")),
        )
    )
    assert a.to_arrow().to_pylist() == [
        [104, 101, 108, 108, 111],
        [116, 104, 101, 114, 101],
    ]

    a = pyarrow.array(
        ["one", "two", "three", "two", "two", "one", "three", "one"]
    ).dictionary_encode()
    b = ak._connect.pyarrow.handle_arrow(a)
    assert isinstance(b, ak.contents.IndexedOptionArray)
    assert to_list(b) == ["one", "two", "three", "two", "two", "one", "three", "one"]

    a = ak.highlevel.Array([[1.1, 2.2, 3.3], [], None, [4.4, 5.5]]).layout
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
        [1.1, 2.2, 3.3],
        [],
        None,
        [4.4, 5.5],
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, False, False, True, True, False, False])),
        ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 999, 314, 4.4, 5.5])),
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == [1.1, 2.2, 3.3, None, None, 4.4, 5.5]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8(np.array([False, False, False, True, True, False, False])),
        ak.operations.from_iter(
            [b"hello", b"", b"there", b"yuk", b"", b"o", b"hellothere"]
        ).layout,
        valid_when=False,
    )
    assert a.to_arrow().to_pylist() == [
        b"hello",
        b"",
        b"there",
        None,
        None,
        b"o",
        b"hellothere",
    ]

    a = ak.contents.ByteMaskedArray(
        ak.index.Index8([True, True, False, True]),
        ak.operations.from_iter([[1.1, 2.2, 3.3], [], [999], [4.4, 5.5]]).layout,
        valid_when=True,
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
        [1.1, 2.2, 3.3],
        [],
        None,
        [4.4, 5.5],
    ]

    a = ak.operations.from_iter([[1, 2, 3], [], [4, 5], 999, 123]).layout
    assert a.to_arrow().to_pylist() == [[1, 2, 3], [], [4, 5], 999, 123]
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
        [1, 2, 3],
        [],
        [4, 5],
        999,
        123,
    ]


def test_arrow_coverage100_broken_unions():
    a = ak.operations.from_iter([[1, 2, 3], [], [4, 5], 999, 123]).layout
    b = ak.contents.ByteMaskedArray.simplified(
        ak.index.Index8(np.array([True, True, False, False, True])),
        a,
        valid_when=True,
    )
    assert b.to_arrow().to_pylist() == [[1, 2, 3], [], None, None, 123]
    assert to_list(ak._connect.pyarrow.handle_arrow(b.to_arrow())) == [
        [1, 2, 3],
        [],
        None,
        None,
        123,
    ]

    content1 = ak.operations.from_iter([1.1, 2.2, 3.3, 4.4, 5.5]).layout
    content2 = ak.operations.from_iter(["hello"]).layout[1:]
    a = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 0, 0, 0, 0], "i1")),
        ak.index.Index32(np.array([0, 1, 2, 3, 4], "i4")),
        [content1, content2],
    )
    assert to_list(a) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert a.to_arrow().to_pylist() == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert to_list(ak._connect.pyarrow.handle_arrow(a.to_arrow())) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]

    a = pyarrow.UnionArray.from_sparse(
        pyarrow.array([0, 0, 0, 0, 0], type=pyarrow.int8()),
        [
            pyarrow.array([0.0, 1.1, None, 3.3, 4.4]),
            pyarrow.array([True, None, False, True, False]),
        ],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        0.0,
        1.1,
        None,
        3.3,
        4.4,
    ]

    a = pyarrow.UnionArray.from_sparse(
        pyarrow.array([0, 1, 0, 1, 1], "i1"),
        [
            pyarrow.array([[0.0, 1.1, 2.2], [], None, [5.5], [6.6, 7.7, 8.8, 9.9]]),
            pyarrow.array([0.0, 1.1, 2.2, None, None]),
        ],
        ["0", "1"],
        [0, 1],
    )
    assert a.to_pylist() == [[0.0, 1.1, 2.2], 1.1, None, None, None]
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        [0.0, 1.1, 2.2],
        1.1,
        None,
        None,
        None,
    ]

    a = pyarrow.chunked_array([pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5])])
    assert to_list(ak._connect.pyarrow.handle_arrow(a)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]


# NumpyArray in Awkward Arrays translate to their corresponding DataType Arrays in Arrow
def test_nonzero_offset_fromarrow_NumpyArray_1():
    boolarray = ak.contents.NumpyArray(
        np.array([True, True, True, False, False, True, False, True, False, True])
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(boolarray.to_arrow()[5:])
    ) == pyarrow.Array.to_pylist(boolarray.to_arrow()[5:])


def test_nonzero_offset_fromarrow_NumpyArray_2():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(content.to_arrow()[2:])
    ) == pyarrow.Array.to_pylist(content.to_arrow()[2:])


def test_nonzero_offset_fromarrow_NumpyArray_3():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(content.to_arrow()[2:5])
    ) == pyarrow.Array.to_pylist(content.to_arrow()[2:5])


def test_nonzero_offset_fromarrow_NumpyArray_4():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(content.to_arrow()[0:9:2])
    ) == pyarrow.Array.to_pylist(content.to_arrow()[0:9:2])


def test_nonzero_offset_fromarrow_NumpyArray_5():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(content.to_arrow()[-2:10])
    ) == pyarrow.Array.to_pylist(content.to_arrow()[-2:10])


def test_nonzero_offset_fromarrow_NumpyArray_6():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(content.to_arrow()[-3:3:-1])
    ) == pyarrow.Array.to_pylist(content.to_arrow()[-3:3:-1])


# ListOffsetArrays in Awkward Arrays translate to ListArrays in Arrow
def test_nonzero_offset_fromarrow_ListOffsetArray_1():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(listoffsetarray.to_arrow()[2:])
    ) == pyarrow.Array.to_pylist(listoffsetarray.to_arrow()[2:])


def test_nonzero_offset_fromarrow_ListOffsetArray_2():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(listoffsetarray.to_arrow()[2:5])
    ) == pyarrow.Array.to_pylist(listoffsetarray.to_arrow()[2:5])


def test_nonzero_offset_fromarrow_ListOffsetArray_3():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(listoffsetarray.to_arrow()[0:5:2])
    ) == pyarrow.Array.to_pylist(listoffsetarray.to_arrow()[0:5:2])


def test_nonzero_offset_fromarrow_ListOffsetArray_4():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(listoffsetarray.to_arrow()[-3:3:-1])
    ) == pyarrow.Array.to_pylist(listoffsetarray.to_arrow()[-3:3:-1])


# RegularArrays in Awkward Arrays translate to ListArrays in Arrow
def test_nonzero_offset_fromarrow_RegularArray_1():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(regulararray.to_arrow()[2:])
    ) == pyarrow.Array.to_pylist(regulararray.to_arrow()[2:])


def test_nonzero_offset_fromarrow_RegularArray_2():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(regulararray.to_arrow()[2:5])
    ) == pyarrow.Array.to_pylist(regulararray.to_arrow()[2:5])


def test_nonzero_offset_fromarrow_RegularArray_3():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(regulararray.to_arrow()[0:5:2])
    ) == pyarrow.Array.to_pylist(regulararray.to_arrow()[0:5:2])


def test_nonzero_offset_fromarrow_RegularArray_4():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    regulararray = ak.contents.RegularArray(listoffsetarray, 2, zeros_length=0)
    assert to_list(
        ak._connect.pyarrow.handle_arrow(regulararray.to_arrow()[-3:3:-1])
    ) == pyarrow.Array.to_pylist(regulararray.to_arrow()[-3:3:-1])


# RecordArrays in Awkward Arrays translate to Struct Arrays in Arrow
def test_nonzero_offset_fromarrow_RecordArray_1():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "chonks", "2", "wonky"],
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(recordarray.to_arrow()[2:])
    ) == pyarrow.Array.to_pylist(recordarray.to_arrow()[2:])


def test_nonzero_offset_fromarrow_RecordArray_2():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "chonks", "2", "wonky"],
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(recordarray.to_arrow()[2:5])
    ) == pyarrow.Array.to_pylist(recordarray.to_arrow()[2:5])


def test_nonzero_offset_fromarrow_RecordArray_3():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "chonks", "2", "wonky"],
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(recordarray.to_arrow()[0:5:2])
    ) == pyarrow.Array.to_pylist(recordarray.to_arrow()[0:5:2])


def test_nonzero_offset_fromarrow_RecordArray_4():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "chonks", "2", "wonky"],
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(recordarray.to_arrow()[-3:3:-1])
    ) == pyarrow.Array.to_pylist(recordarray.to_arrow()[-3:3:-1])


def test_nonzero_offset_fromarrow_RecordArray_4_again():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 10, 10]))
    listoffsetarray = ak.contents.ListOffsetArray(offsets, content)

    content1 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5]))
    content2 = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index32(np.array([0, 3, 3, 5, 6, 9]))
    recordarray = ak.contents.RecordArray(
        [content1, listoffsetarray, content2, content1],
        fields=["one", "chonks", "2", "wonky"],
    )
    assert to_list(
        ak._connect.pyarrow.handle_arrow(recordarray.to_arrow()[-3:3:-1])
    ) == pyarrow.Array.to_pylist(recordarray.to_arrow()[-3:3:-1])


def test_nonzero_offset_fromarrow_UnionArray_1():
    content0 = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.contents.UnionArray(tags, index, [content0, content])
    assert to_list(
        ak._connect.pyarrow.handle_arrow(array.to_arrow()[2:])
    ) == pyarrow.Array.to_pylist(array.to_arrow()[2:])


def test_nonzero_offset_fromarrow_UnionArray_2():
    content0 = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.contents.UnionArray(tags, index, [content0, content])
    assert to_list(
        ak._connect.pyarrow.handle_arrow(array.to_arrow()[2:5])
    ) == pyarrow.Array.to_pylist(array.to_arrow()[2:5])


def test_nonzero_offset_fromarrow_UnionArray_3():
    content0 = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).layout
    content = ak.highlevel.Array(
        ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    ).layout
    tags = ak.index.Index8(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.int8))
    index = ak.index.Index32(np.array([0, 1, 0, 1, 2, 2, 4, 3], dtype=np.int32))
    array = ak.contents.UnionArray(tags, index, [content0, content])
    assert to_list(
        ak._connect.pyarrow.handle_arrow(array.to_arrow()[0:5:1])
    ) == pyarrow.Array.to_pylist(array.to_arrow()[0:5:1])


def test_nonzero_offset_fromarrow_ArrowDictionaryArray_1():
    a = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]),
        pyarrow.array(["one", None, "three"]),
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[2:])) == [
        "three",
        "three",
        None,
        "one",
        "three",
        None,
        None,
    ]


def test_nonzero_offset_fromarrow_ArrowDictionaryArray_2():
    a = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]),
        pyarrow.array(["one", None, "three"]),
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[2:5])) == [
        "three",
        "three",
        None,
    ]


def test_nonzero_offset_fromarrow_ArrowDictionaryArray_3():
    a = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]),
        pyarrow.array(["one", None, "three"]),
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[0:8:2])) == [
        "one",
        "three",
        None,
        "three",
    ]


def test_nonzero_offset_fromarrow_ArrowDictionaryArray_4():
    a = pyarrow.DictionaryArray.from_arrays(
        pyarrow.array([0, 0, 2, 2, 1, 0, 2, 1, 1]),
        pyarrow.array(["one", None, "three"]),
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[-3:3:-1])) == [
        "three",
        "one",
        None,
    ]


def test_nonzero_offset_fromarrow_ArrowRecordBatch_1():
    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            pyarrow.array([[1, 2, 3], [], [], [4, 5], [6]]),
        ],
        ["a", "b"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[0])) == a[0].to_pylist()


def test_nonzero_offset_fromarrow_ArrowRecordBatch_2():
    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            pyarrow.array([[1, 2, 3], [], [], [4, 5], [6]]),
        ],
        ["a", "b"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[2:])) == [
        {"a": 3.3, "b": []},
        {"a": 4.4, "b": [4, 5]},
        {"a": 5.5, "b": [6]},
    ]


def test_nonzero_offset_fromarrow_ArrowRecordBatch_3():
    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            pyarrow.array([[1, 2, 3], [], [], [4, 5], [6]]),
        ],
        ["a", "b"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[2:5])) == [
        {"a": 3.3, "b": []},
        {"a": 4.4, "b": [4, 5]},
        {"a": 5.5, "b": [6]},
    ]


def test_nonzero_offset_fromarrow_ArrowRecordBatch_4():
    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            pyarrow.array([[1, 2, 3], [], [], [4, 5], [6]]),
        ],
        ["a", "b"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[0:5:2])) == [
        {"a": 1.1, "b": [1, 2, 3]},
        {"a": 3.3, "b": []},
        {"a": 5.5, "b": [6]},
    ]


def test_nonzero_offset_fromarrow_ArrowRecordBatch_4_again():
    a = pyarrow.RecordBatch.from_arrays(
        [
            pyarrow.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            pyarrow.array([[1, 2, 3], [], [], [4, 5], [6]]),
        ],
        ["a", "b"],
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[-2:0:-1])) == [
        {"a": 4.4, "b": [4, 5]},
        {"a": 3.3, "b": []},
        {"a": 2.2, "b": []},
    ]


def test_nonzero_offset_fromarrow_ArrowTable_1():
    a = pyarrow.Table.from_batches(
        [
            pyarrow.RecordBatch.from_arrays(
                [
                    pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                    pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            {"x": 2, "y": 2.2},
                            {"x": 3, "y": 3.3},
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            None,
                            None,
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            [
                                {"x": 1, "y": 1.1},
                                {"x": 2, "y": 2.2},
                                {"x": 3, "y": 3.3},
                            ],
                            [],
                            [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
                            [None],
                            [{"x": 6, "y": 6.6}],
                        ]
                    ),
                ],
                ["a", "b", "c", "d", "e"],
            ),
            pyarrow.RecordBatch.from_arrays(
                [
                    pyarrow.array([1.1, 2.2, 3.3, None, 5.5]),
                    pyarrow.array([[1, 2, 3], [], [4, 5], [None], [6]]),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            {"x": 2, "y": 2.2},
                            {"x": 3, "y": 3.3},
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            {"x": 1, "y": 1.1},
                            None,
                            None,
                            {"x": 4, "y": None},
                            {"x": 5, "y": 5.5},
                        ]
                    ),
                    pyarrow.array(
                        [
                            [
                                {"x": 1, "y": 1.1},
                                {"x": 2, "y": 2.2},
                                {"x": 3, "y": 3.3},
                            ],
                            [],
                            [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
                            [None],
                            [{"x": 6, "y": 6.6}],
                        ]
                    ),
                ],
                ["a", "b", "c", "d", "e"],
            ),
        ]
    )
    assert to_list(ak._connect.pyarrow.handle_arrow(a[0:5:2])) == [
        {
            "a": 1.1,
            "b": [1, 2, 3],
            "c": {"x": 1, "y": 1.1},
            "d": {"x": 1, "y": 1.1},
            "e": [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}],
        },
        {
            "a": 3.3,
            "b": [4, 5],
            "c": {"x": 3, "y": 3.3},
            "d": None,
            "e": [{"x": 4, "y": None}, {"x": 5, "y": 5.5}],
        },
        {
            "a": 5.5,
            "b": [6],
            "c": {"x": 5, "y": 5.5},
            "d": {"x": 5, "y": 5.5},
            "e": [{"x": 6, "y": 6.6}],
        },
    ]
