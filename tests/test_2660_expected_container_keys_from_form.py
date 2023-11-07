# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak


def test_EmptyArray():
    a = ak.contents.emptyarray.EmptyArray()
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_NumpyArray_to_RegularArray():
    a = ak.operations.from_numpy(np.arange(2 * 3 * 5).reshape(2, 3, 5)).layout
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    b = a.to_RegularArray()
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    a = ak.operations.from_numpy(np.arange(2 * 0 * 5).reshape(2, 0, 5)).layout
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    b = a.to_RegularArray()
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_NumpyArray():
    a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    b = ak.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_RegularArray_NumpyArray():
    a = ak.contents.regulararray.RegularArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        ),
        3,
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    b = ak.contents.regulararray.RegularArray(
        ak.contents.emptyarray.EmptyArray(), 0, zeros_length=10
    )
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_ListArray_NumpyArray():
    # 200 is inaccessible in stops
    # 6.6, 7.7, and 8.8 are inaccessible in content
    a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], dtype=np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], dtype=np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_ListOffsetArray_NumpyArray():
    # 6.6 and 7.7 are inaccessible
    a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6])),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])
        ),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_RecordArray_NumpyArray():
    # 5.5 is inaccessible
    a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4])),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 5.5 is inaccessible
    b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4])),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    c = ak.contents.recordarray.RecordArray([], [], 10)
    form, length, container = ak.to_buffers(c)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    d = ak.contents.recordarray.RecordArray([], None, 10)
    form, length, container = ak.to_buffers(d)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_IndexedArray_NumpyArray():
    # 4.4 is inaccessible; 3.3 and 5.5 appear twice
    a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_IndexedOptionArray_NumpyArray():
    # 1.1 and 4.4 are inaccessible; 3.3 appears twice
    a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4])),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_ByteMaskedArray_NumpyArray():
    # 2.2, 4.4, and 6.6 are inaccessible
    a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 2.2, 4.4, and 6.6 are inaccessible
    b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_BitMaskedArray_NumpyArray():
    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    b = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    c = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )
    form, length, container = ak.to_buffers(c)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    d = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.numpyarray.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )
    form, length, container = ak.to_buffers(d)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_UnmaskedArray_NumpyArray():
    a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_UnionArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    a = ak.contents.unionarray.UnionArray.simplified(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3])),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_RegularArray_RecordArray_NumpyArray():
    # 6.6 is inaccessible
    a = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        3,
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    b = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_ListArray_RecordArray_NumpyArray():
    # 200 is inaccessible in stops
    # 6.6, 7.7, and 8.8 are inaccessible in content
    a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1])),
        ak.index.Index(np.array([7, 100, 3, 200])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_ListOffsetArray_RecordArray_NumpyArray():
    # 6.6 and 7.7 are inaccessible
    a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])
                )
            ],
            ["nest"],
        ),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_IndexedArray_RecordArray_NumpyArray():
    # 4.4 is inaccessible; 3.3 and 5.5 appear twice
    a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_IndexedOptionArray_RecordArray_NumpyArray():
    # 1.1 and 4.4 are inaccessible; 3.3 appears twice
    a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4])),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_ByteMaskedArray_RecordArray_NumpyArray():
    # 2.2, 4.4, and 6.6 are inaccessible
    a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=True,
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 2.2, 4.4, and 6.6 are inaccessible
    b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
        valid_when=False,
    )
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_BitMaskedArray_RecordArray_NumpyArray():
    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    a = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
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
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    b = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
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
    form, length, container = ak.to_buffers(b)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    c = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
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
    form, length, container = ak.to_buffers(c)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype

    # 4.0, 5.0, 6.0, 7.0, 2.2, 4.4, and 6.6 are inaccessible
    d = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(
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
                    dtype=np.uint8,
                )
            )
        ),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
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
    form, length, container = ak.to_buffers(d)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_UnmaskedArray_RecordArray_NumpyArray():
    a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
                )
            ],
            ["nest"],
        )
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype


def test_UnionArray_RecordArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    a = ak.contents.unionarray.UnionArray.simplified(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.recordarray.RecordArray(
                [ak.contents.numpyarray.NumpyArray(np.array([1, 2, 3]))], ["nest"]
            ),
            ak.contents.recordarray.RecordArray(
                [
                    ak.contents.numpyarray.NumpyArray(
                        np.array([1.1, 2.2, 3.3, 4.4, 5.5])
                    )
                ],
                ["nest"],
            ),
        ],
    )
    form, length, container = ak.to_buffers(a)
    for name, dtype in form.expected_from_buffers().items():
        assert container[name].dtype == dtype
