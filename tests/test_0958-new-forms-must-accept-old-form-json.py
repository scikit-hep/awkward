# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

# flake8: noqa


def test_EmptyArray():
    a = ak.layout.EmptyArray()


def test_NumpyArray():
    a = ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))

    b = ak.layout.NumpyArray(np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5))


def test_RegularArray_NumpyArray():
    a = ak.layout.RegularArray(
        ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        3,
    )

    b = ak.layout.RegularArray(ak.layout.EmptyArray(), 0, zeros_length=10)


def test_ListArray_NumpyArray():
    a = ak.layout.ListArray64(
        ak.layout.Index64(np.array([4, 100, 1], dtype=np.int64)),
        ak.layout.Index64(np.array([7, 100, 3, 200], dtype=np.int64)),
        ak.layout.NumpyArray(np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])),
    )


def test_ListOffsetArray_NumpyArray():
    a = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([1, 4, 4, 6], dtype=np.int64)),
        ak.layout.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )


def test_RecordArray_NumpyArray():
    a = ak.layout.RecordArray(
        [
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
            ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )

    b = ak.layout.RecordArray(
        [
            ak.layout.NumpyArray(np.array([0, 1, 2, 3, 4], dtype=np.int64)),
            ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )

    c = ak.layout.RecordArray([], [], 10)

    d = ak.layout.RecordArray([], None, 10)


def test_IndexedArray_NumpyArray():
    a = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([2, 2, 0, 1, 4, 5, 4], dtype=np.int64)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )


def test_IndexedOptionArray_NumpyArray():
    a = ak.layout.IndexedOptionArray64(
        ak.layout.Index64(np.array([2, 2, -1, 1, -1, 5, 4], dtype=np.int64)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )


def test_ByteMaskedArray_NumpyArray():
    a = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    b = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )


def test_BitMaskedArray_NumpyArray():
    a = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    b = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=False,
    )

    c = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=True,
        length=13,
        lsb_order=True,
    )

    d = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.NumpyArray(
            np.array(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
            )
        ),
        valid_when=False,
        length=13,
        lsb_order=True,
    )


def test_UnmaskedArray_NumpyArray():
    a = ak.layout.UnmaskedArray(ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3])))


def test_UnionArray_NumpyArray():
    a = ak.layout.UnionArray8_64(
        ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.Index64(np.array([4, 3, 0, 1, 2, 2, 4, 100], dtype=np.int64)),
        [
            ak.layout.NumpyArray(np.array([1, 2, 3], dtype=np.int64)),
            ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )


def test_RegularArray_RecordArray_NumpyArray():
    a = ak.layout.RegularArray(
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
        3,
    )

    b = ak.layout.RegularArray(
        ak.layout.RecordArray([ak.layout.EmptyArray()], ["nest"]),
        0,
        zeros_length=10,
    )


def test_ListArray_RecordArray_NumpyArray():
    a = ak.layout.ListArray64(
        ak.layout.Index64(np.array([4, 100, 1], dtype=np.int64)),
        ak.layout.Index64(np.array([7, 100, 3, 200], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8]))],
            ["nest"],
        ),
    )


def test_ListOffsetArray_RecordArray_NumpyArray():
    a = ak.layout.ListOffsetArray64(
        ak.layout.Index64(np.array([1, 4, 4, 6], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])],
            ["nest"],
        ),
    )


def test_IndexedArray_RecordArray_NumpyArray():
    a = ak.layout.IndexedArray64(
        ak.layout.Index64(np.array([2, 2, 0, 1, 4, 5, 4], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
    )


def test_IndexedOptionArray_RecordArray_NumpyArray():
    a = ak.layout.IndexedOptionArray64(
        ak.layout.Index64(np.array([2, 2, -1, 1, -1, 5, 4], dtype=np.int64)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
    )


def test_ByteMaskedArray_RecordArray_NumpyArray():
    a = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
        valid_when=True,
    )

    b = ak.layout.ByteMaskedArray(
        ak.layout.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]))],
            ["nest"],
        ),
        valid_when=False,
    )


def test_BitMaskedArray_RecordArray_NumpyArray():
    a = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
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

    b = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
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

    c = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
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

    d = ak.layout.BitMaskedArray(
        ak.layout.IndexU8(
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
        ak.layout.RecordArray(
            [
                ak.layout.NumpyArray(
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


def test_UnmaskedArray_RecordArray_NumpyArray():
    a = ak.layout.UnmaskedArray(
        ak.layout.RecordArray(
            [ak.layout.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))],
            ["nest"],
        )
    )


def test_UnionArray_RecordArray_NumpyArray():
    a = ak.layout.UnionArray8_64(
        ak.layout.Index8(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.layout.Index64(np.array([4, 3, 0, 1, 2, 2, 4, 100], dtype=np.int64)),
        [
            ak.layout.RecordArray(
                [ak.layout.NumpyArray(np.array([1, 2, 3], dtype=np.int64))], ["nest"]
            ),
            ak.layout.RecordArray(
                [ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))],
                ["nest"],
            ),
        ],
    )
