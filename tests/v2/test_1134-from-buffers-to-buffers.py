# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

ak_to_buffers = ak._v2.operations.to_buffers
ak_from_buffers = ak._v2.operations.from_buffers
ak_from_iter = ak._v2.operations.from_iter

to_list = ak._v2.operations.to_list


def test_EmptyArray():
    v2a = ak._v2.contents.emptyarray.EmptyArray()
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_NumpyArray():
    v2a = ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)


def test_RegularArray_NumpyArray():
    v2a = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.emptyarray.EmptyArray(), 0, zeros_length=10
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)


def test_ListArray_NumpyArray():
    v2a = ak._v2.contents.listarray.ListArray(
        ak._v2.index.Index(np.array([4, 100, 1], np.int64)),
        ak._v2.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_ListOffsetArray_NumpyArray():
    v2a = ak._v2.contents.listoffsetarray.ListOffsetArray(
        ak._v2.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_RecordArray_NumpyArray():
    v2a = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        ["x", "y"],
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        None,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)

    v2c = ak._v2.contents.recordarray.RecordArray([], [], 10)
    assert to_list(ak_from_buffers(*ak_to_buffers(v2c))) == to_list(v2c)

    v2d = ak._v2.contents.recordarray.RecordArray([], None, 10)
    assert to_list(ak_from_buffers(*ak_to_buffers(v2d))) == to_list(v2d)


def test_IndexedArray_NumpyArray():
    v2a = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_IndexedOptionArray_NumpyArray():
    v2a = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_ByteMaskedArray_NumpyArray():
    v2a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)


def test_BitMaskedArray_NumpyArray():
    v2a = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)

    v2c = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2c))) == to_list(v2c)

    v2d = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2d))) == to_list(v2d)


def test_UnmaskedArray_NumpyArray():
    v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_UnionArray_NumpyArray():
    v2a = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_RegularArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.recordarray.RecordArray(
            [ak._v2.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)


def test_ListArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.listarray.ListArray(
        ak._v2.index.Index(np.array([4, 100, 1], np.int64)),
        ak._v2.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_ListOffsetArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.listoffsetarray.ListOffsetArray(
        ak._v2.index.Index(np.array([1, 4, 4, 6], np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    [6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]
                )
            ],
            ["nest"],
        ),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_IndexedArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_IndexedOptionArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_ByteMaskedArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=False,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)


def test_BitMaskedArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)

    v2b = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2b))) == to_list(v2b)

    v2c = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2c))) == to_list(v2c)

    v2d = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(
            np.packbits(
                np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                    ],
                    np.uint8,
                )
            )
        ),
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.numpyarray.NumpyArray(
                    np.array(
                        [
                            0.0,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                            6.0,
                            7.0,
                            1.1,
                            2.2,
                            3.3,
                            4.4,
                            5.5,
                            6.6,
                        ]
                    )
                )
            ],
            ["nest"],
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2d))) == to_list(v2d)


def test_UnmaskedArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.recordarray.RecordArray(
            [ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))],
            ["nest"],
        )
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_UnionArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak._v2.contents.recordarray.RecordArray(
                [ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3], np.int64))],
                ["nest"],
            ),
            ak._v2.contents.recordarray.RecordArray(
                [
                    ak._v2.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )
    assert to_list(ak_from_buffers(*ak_to_buffers(v2a))) == to_list(v2a)


def test_fromiter():
    # Float64Builder
    assert to_list(ak_from_iter([1.1, 2.2, 3.3])) == [1.1, 2.2, 3.3]

    # BoolBuilder
    assert to_list(ak_from_iter([True, False, True])) == [True, False, True]

    # Complex128Builder
    assert to_list(ak_from_iter([1, 2 + 2j, 3j])) == [1, 2 + 2j, 3j]

    # DatetimeBuilder
    assert to_list(ak_from_iter([np.datetime64("2021-11-08T01:02:03")])) == [
        np.datetime64("2021-11-08T01:02:03")
    ]

    # Int64Builder
    assert to_list(ak_from_iter([1, 2, 3])) == [1, 2, 3]

    # ListBuilder
    assert to_list(ak_from_iter([[1, 2, 3], [], [4, 5]])) == [[1, 2, 3], [], [4, 5]]

    # OptionBuilder
    assert to_list(ak_from_iter([1, 2, None, 3])) == [1, 2, None, 3]

    # StringBuilder
    assert to_list(ak_from_iter(["hello", "there"])) == ["hello", "there"]

    # TupleBuilder
    assert to_list(ak_from_iter([(1, 1.1), (2, 2.2)])) == [(1, 1.1), (2, 2.2)]

    # RecordBuilder
    assert to_list(ak_from_iter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
    ]

    # UnionBuilder
    assert to_list(ak_from_iter([1, 2, [1, 2, 3]])) == [1, 2, [1, 2, 3]]

    # UnknownBuilder
    assert to_list(ak_from_iter([])) == []
