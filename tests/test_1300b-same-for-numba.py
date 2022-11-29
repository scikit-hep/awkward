# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")

ak.numba.register_and_check()


def test_NumpyArray():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3]),
        parameters={"some": "stuff", "other": [1, 2, "three"]},
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[1]
        out[2] = obj[3]

    out = np.zeros(3, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_EmptyArray():
    v2a = ak.contents.emptyarray.EmptyArray()

    @numba.njit
    def f(obj):
        return len(obj)

    assert f(ak.highlevel.Array(v2a)) == 0


def test_NumpyArray_shape():
    v2a = ak.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = len(obj[0][0])
        out[3] = obj[0][0][0]
        out[4] = obj[0][0][1]
        out[5] = obj[0][1][0]
        out[6] = obj[0][1][1]
        out[7] = obj[1][0][0]
        out[8] = obj[1][1][1]

    out = np.zeros(9, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [2.0, 3.0, 5.0, 0.0, 1.0, 5.0, 6.0, 15.0, 21.0]


def test_RegularArray_NumpyArray():
    v2a = ak.contents.regulararray.RegularArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0][0]
        out[2] = obj[0][1]
        out[3] = obj[1][0]
        out[4] = obj[1][1]
        out[5] = len(obj[1])

    out = np.zeros(6, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [2.0, 0.0, 1.1, 3.3, 4.4, 3.0]

    v2b = ak.contents.regulararray.RegularArray(
        ak.contents.emptyarray.EmptyArray().to_NumpyArray(np.dtype(np.float64)),
        0,
        zeros_length=10,
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = len(obj[1])

    out = np.zeros(3, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [10.0, 0.0, 0.0]


def test_ListArray_NumpyArray():
    v2a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = obj[0][0]
        out[3] = obj[0][1]
        out[4] = obj[0][2]
        out[5] = len(obj[1])
        out[6] = len(obj[2])
        out[7] = obj[2][0]
        out[8] = obj[2][1]

    out = np.zeros(9, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [3.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5]


def test_ListOffsetArray_NumpyArray():
    v2a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = obj[0][0]
        out[3] = obj[0][1]
        out[4] = obj[0][2]
        out[5] = len(obj[1])
        out[6] = len(obj[2])
        out[7] = obj[2][0]
        out[8] = obj[2][1]
        out[9] = len(obj[3])
        out[10] = obj[3][0]

    out = np.zeros(11, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [4.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5, 1.0, 7.7]


def test_RecordArray_NumpyArray():
    v2a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
        parameters={"__record__": "Something"},
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        rec1 = obj[1]
        rec4 = obj[4]
        out[1] = rec1.x
        out[2] = rec1.y
        out[3] = rec4.x
        out[4] = rec4.y

    out = np.zeros(5, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        rec1 = obj[1]
        rec4 = obj[4]
        out[1] = rec1["0"]
        out[2] = rec1["1"]
        out[3] = rec4["0"]
        out[4] = rec4["1"]

    out = np.zeros(5, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2c = ak.contents.recordarray.RecordArray([], [], 10)

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        obj[5]

    out = np.zeros(1, dtype=np.float64)
    f(out, ak.highlevel.Array(v2c))
    assert out.tolist() == [10.0]

    v2d = ak.contents.recordarray.RecordArray([], None, 10)

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        obj[5]

    out = np.zeros(1, dtype=np.float64)
    f(out, ak.highlevel.Array(v2d))
    assert out.tolist() == [10.0]


def test_IndexedArray_NumpyArray():
    v2a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0]
        out[2] = obj[1]
        out[3] = obj[2]
        out[4] = obj[3]
        out[5] = obj[4]
        out[6] = obj[5]
        out[7] = obj[6]

    out = np.zeros(8, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [7.0, 2.2, 2.2, 0.0, 1.1, 4.4, 5.5, 4.4]


def test_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0

    out = np.zeros(8, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [7.0, 2.2, 2.2, 999.0, 1.1, 999.0, 5.5, 4.4]


def test_ByteMaskedArray_NumpyArray():
    v2a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0

    out = np.zeros(6, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]

    v2b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0

    out = np.zeros(6, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]


def test_BitMaskedArray_NumpyArray():
    v2a = ak.contents.bitmaskedarray.BitMaskedArray(
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
                    np.uint8,
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

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2b = ak.contents.bitmaskedarray.BitMaskedArray(
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
                    np.uint8,
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

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2c = ak.contents.bitmaskedarray.BitMaskedArray(
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
                    np.uint8,
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

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2c))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2d = ak.contents.bitmaskedarray.BitMaskedArray(
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
                    np.uint8,
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

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2d))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]


def test_UnmaskedArray_NumpyArray():
    v2a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )

    @numba.njit
    def f(out, obj):
        out[0] = len(obj)
        out[1] = obj[1] if obj[1] is not None else 999.0
        out[2] = obj[3] if obj[3] is not None else 999.0

    out = np.zeros(3, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_nested_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([999.0, 0.0, 1.1, 2.2, 3.3]),
            parameters={"some": "stuff", "other": [1, 2, "three"]},
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[1]
        out[2] = obj[3]

    out = np.zeros(3, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [4.0, 1.1, 3.3]


def test_nested_NumpyArray_shape():
    data = np.full((3, 3, 5), 999, dtype=np.int64)
    data[1:3] = np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)

    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
        ak.contents.numpyarray.NumpyArray(data),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = len(obj[0][0])
        out[3] = obj[0][0][0]
        out[4] = obj[0][0][1]
        out[5] = obj[0][1][0]
        out[6] = obj[0][1][1]
        out[7] = obj[1][0][0]
        out[8] = obj[1][1][1]

    out = np.zeros(9, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [2.0, 3.0, 5.0, 0.0, 1.0, 5.0, 6.0, 15.0, 21.0]


def test_nested_RegularArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3], dtype=np.int64)),
        ak.contents.regulararray.RegularArray(
            ak.contents.numpyarray.NumpyArray(
                np.array([999, 999, 999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
            3,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0][0]
        out[2] = obj[0][1]
        out[3] = obj[1][0]
        out[4] = obj[1][1]
        out[5] = len(obj[1])

    out = np.zeros(6, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [2.0, 0.0, 1.1, 3.3, 4.4, 3.0]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.regulararray.RegularArray(
            ak.contents.emptyarray.EmptyArray().to_NumpyArray(np.dtype(np.float64)),
            0,
            zeros_length=11,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = len(obj[1])

    out = np.zeros(3, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [10.0, 0.0, 0.0]


def test_nested_ListArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 4], dtype=np.int64)),
        ak.contents.listarray.ListArray(
            ak.index.Index(np.array([999, 4, 100, 1], np.int64)),
            ak.index.Index(np.array([999, 7, 100, 3, 200], np.int64)),
            ak.contents.numpyarray.NumpyArray(
                np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
            ),
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = obj[0][0]
        out[3] = obj[0][1]
        out[4] = obj[0][2]
        out[5] = len(obj[1])
        out[6] = len(obj[2])
        out[7] = obj[2][0]
        out[8] = obj[2][1]

    out = np.zeros(9, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [3.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5]


def test_nested_ListOffsetArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.listoffsetarray.ListOffsetArray(
            ak.index.Index(np.array([1, 1, 4, 4, 6, 7], np.int64)),
            ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = len(obj[0])
        out[2] = obj[0][0]
        out[3] = obj[0][1]
        out[4] = obj[0][2]
        out[5] = len(obj[1])
        out[6] = len(obj[2])
        out[7] = obj[2][0]
        out[8] = obj[2][1]
        out[9] = len(obj[3])
        out[10] = obj[3][0]

    out = np.zeros(11, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [4.0, 3.0, 1.1, 2.2, 3.3, 0.0, 2.0, 4.4, 5.5, 1.0, 7.7]


def test_nested_RecordArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0, 1, 2, 3, 4], np.int64)
                ),
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
                ),
            ],
            ["x", "y"],
            parameters={"__record__": "Something"},
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        rec1 = obj[1]
        rec4 = obj[4]
        out[1] = rec1.x
        out[2] = rec1.y
        out[3] = rec4.x
        out[4] = rec4.y

    out = np.zeros(5, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0, 1, 2, 3, 4], np.int64)
                ),
                ak.contents.numpyarray.NumpyArray(
                    np.array([999, 0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
                ),
            ],
            None,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        rec1 = obj[1]
        rec4 = obj[4]
        out[1] = rec1["0"]
        out[2] = rec1["1"]
        out[3] = rec4["0"]
        out[4] = rec4["1"]

    out = np.zeros(5, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [5.0, 1, 1.1, 4, 4.4]

    v2c = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.recordarray.RecordArray([], [], 11),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        obj[5]

    out = np.zeros(1, dtype=np.float64)
    f(out, ak.highlevel.Array(v2c))
    assert out.tolist() == [10.0]

    v2d = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 11], dtype=np.int64)),
        ak.contents.recordarray.RecordArray([], None, 11),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        obj[5]

    out = np.zeros(1, dtype=np.float64)
    f(out, ak.highlevel.Array(v2d))
    assert out.tolist() == [10.0]


def test_nested_IndexedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak.contents.indexedarray.IndexedArray(
            ak.index.Index(np.array([999, 2, 2, 0, 1, 4, 5, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0]
        out[2] = obj[1]
        out[3] = obj[2]
        out[4] = obj[3]
        out[5] = obj[4]
        out[6] = obj[5]
        out[7] = obj[6]

    out = np.zeros(8, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [7.0, 2.2, 2.2, 0.0, 1.1, 4.4, 5.5, 4.4]


def test_nested_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 8], dtype=np.int64)),
        ak.contents.indexedoptionarray.IndexedOptionArray(
            ak.index.Index(np.array([999, 2, 2, -1, 1, -1, 5, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0

    out = np.zeros(8, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [7.0, 2.2, 2.2, 999.0, 1.1, 999.0, 5.5, 4.4]


def test_nested_ByteMaskedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([123, 1, 0, 1, 0, 1], np.int8)),
            ak.contents.numpyarray.NumpyArray(
                np.array([999, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
            ),
            valid_when=True,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0

    out = np.zeros(6, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
        ak.contents.bytemaskedarray.ByteMaskedArray(
            ak.index.Index(np.array([123, 0, 1, 0, 1, 0], np.int8)),
            ak.contents.numpyarray.NumpyArray(
                np.array([999, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
            ),
            valid_when=False,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0

    out = np.zeros(6, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [5.0, 1.1, 999.0, 3.3, 999.0, 5.5]


def test_nested_BitMaskedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
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
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
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
            ),
            valid_when=True,
            length=14,
            lsb_order=False,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2b = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
                            0,
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
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
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
            ),
            valid_when=False,
            length=14,
            lsb_order=False,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2b))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2c = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
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
                            0,
                            0,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
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
            ),
            valid_when=True,
            length=14,
            lsb_order=True,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2c))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]

    v2d = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 14], dtype=np.int64)),
        ak.contents.bitmaskedarray.BitMaskedArray(
            ak.index.Index(
                np.packbits(
                    np.array(
                        [
                            1,
                            1,
                            1,
                            0,
                            0,
                            0,
                            0,
                            1,
                            1,
                            1,
                            0,
                            1,
                            0,
                            1,
                            0,
                            1,
                            1,
                        ],
                        np.uint8,
                    )
                )
            ),
            ak.contents.numpyarray.NumpyArray(
                np.array(
                    [
                        999,
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
            ),
            valid_when=False,
            length=14,
            lsb_order=True,
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[0] if obj[0] is not None else 999.0
        out[2] = obj[1] if obj[1] is not None else 999.0
        out[3] = obj[2] if obj[2] is not None else 999.0
        out[4] = obj[3] if obj[3] is not None else 999.0
        out[5] = obj[4] if obj[4] is not None else 999.0
        out[6] = obj[5] if obj[5] is not None else 999.0
        out[7] = obj[6] if obj[6] is not None else 999.0
        out[8] = obj[7] if obj[7] is not None else 999.0
        out[9] = obj[8] if obj[8] is not None else 999.0
        out[10] = obj[9] if obj[9] is not None else 999.0
        out[11] = obj[10] if obj[10] is not None else 999.0
        out[12] = obj[11] if obj[11] is not None else 999.0
        out[13] = obj[12] if obj[12] is not None else 999.0

    out = np.zeros(14, dtype=np.float64)
    f(out, ak.highlevel.Array(v2d))
    assert out.tolist() == [
        13.0,
        0.0,
        1.0,
        2.0,
        3.0,
        999.0,
        999.0,
        999.0,
        999.0,
        1.1,
        999.0,
        3.3,
        999.0,
        5.5,
    ]


def test_nested_UnmaskedArray_NumpyArray():
    v2a = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 5], dtype=np.int64)),
        ak.contents.unmaskedarray.UnmaskedArray(
            ak.contents.numpyarray.NumpyArray(np.array([999, 0.0, 1.1, 2.2, 3.3]))
        ),
    )

    @numba.njit
    def f(out, array):
        obj = array[1]
        out[0] = len(obj)
        out[1] = obj[1] if obj[1] is not None else 999.0
        out[2] = obj[3] if obj[3] is not None else 999.0

    out = np.zeros(3, dtype=np.float64)
    f(out, ak.highlevel.Array(v2a))
    assert out.tolist() == [4.0, 1.1, 3.3]
