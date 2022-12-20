# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_EmptyArray():
    v2a = ak.contents.emptyarray.EmptyArray()
    with pytest.raises(IndexError):
        v2a[np.array([0, 1], np.int64)]


def test_NumpyArray():
    v2a = ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    resultv2 = v2a[np.array([0, 1, -2], np.int64)]
    assert to_list(resultv2) == [0.0, 1.1, 2.2]
    assert v2a.to_typetracer()[np.array([0, 1, -2], np.int64)].form == resultv2.form

    v2b = ak.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    resultv2 = v2b[np.array([1, 1, 1], np.int64)]
    assert to_list(resultv2) == [
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
        [[15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
    assert v2b.to_typetracer()[np.array([1, 1, 1], np.int64)].form == resultv2.form


def test_RegularArray_NumpyArray():
    v2a = ak.contents.regulararray.RegularArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )
    resultv2 = v2a[np.array([0, 1], np.int64)]
    assert to_list(resultv2) == [[0.0, 1.1, 2.2], [3.3, 4.4, 5.5]]
    assert v2a.to_typetracer()[np.array([0, 1], np.int64)].form == resultv2.form

    v2b = ak.contents.regulararray.RegularArray(
        ak.contents.emptyarray.EmptyArray(), 0, zeros_length=10
    )
    resultv2 = v2b[np.array([0, 0, 0], np.int64)]
    assert to_list(resultv2) == [[], [], []]
    assert v2b.to_typetracer()[np.array([0, 0, 0], np.int64)].form == resultv2.form

    assert to_list(resultv2) == [[], [], []]


def test_ListArray_NumpyArray():
    v2a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )
    resultv2 = v2a[np.array([1, -1], np.int64)]
    assert to_list(resultv2) == [[], [4.4, 5.5]]
    assert v2a.to_typetracer()[np.array([1, -1], np.int64)].form == resultv2.form


def test_ListOffsetArray_NumpyArray():
    v2a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )
    resultv2 = v2a[np.array([1, 2], np.int64)]
    assert to_list(resultv2) == [[], [4.4, 5.5]]
    assert v2a.to_typetracer()[np.array([1, 2], np.int64)].form == resultv2.form


@pytest.mark.skipif(
    ak._util.win,
    reason="unstable dict order. -- on Windows",
)
def test_RecordArray_NumpyArray():
    v2a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )
    resultv2 = v2a[np.array([1, 2], np.int64)]
    assert to_list(resultv2) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]
    assert v2a.to_typetracer()[np.array([1, 2], np.int64)].form == resultv2.form

    v2b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )
    resultv2 = v2b[np.array([0, 1, 2, 3, -1], np.int64)]
    assert to_list(resultv2) == [(0, 0.0), (1, 1.1), (2, 2.2), (3, 3.3), (4, 4.4)]
    assert (
        v2b.to_typetracer()[np.array([0, 1, 2, 3, -1], np.int64)].form == resultv2.form
    )

    v2c = ak.contents.recordarray.RecordArray([], [], 10)
    resultv2 = v2c[np.array([0], np.int64)]
    assert to_list(resultv2) == [{}]
    assert v2c.to_typetracer()[np.array([0], np.int64)].form == resultv2.form

    v2d = ak.contents.recordarray.RecordArray([], None, 10)
    resultv2 = v2d[np.array([0], np.int64)]
    assert to_list(resultv2) == [()]
    assert v2d.to_typetracer()[np.array([0], np.int64)].form == resultv2.form


def test_IndexedArray_NumpyArray():
    v2a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    resultv2 = v2a[np.array([0, 1, 4], np.int64)]
    assert to_list(resultv2) == [3.3, 3.3, 5.5]
    assert v2a.to_typetracer()[np.array([0, 1, 4], np.int64)].form == resultv2.form


def test_IndexedOptionArray_NumpyArray():
    v2a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    resultv2 = v2a[np.array([0, 1, -1], np.int64)]
    assert to_list(resultv2) == [3.3, 3.3, 5.5]
    assert v2a.to_typetracer()[np.array([0, 1, -1], np.int64)].form == resultv2.form


def test_ByteMaskedArray_NumpyArray():
    v2a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    resultv2 = v2a[np.array([0, 1, 2], np.int64)]
    assert to_list(resultv2) == [1.1, None, 3.3]
    assert v2a.to_typetracer()[np.array([0, 1, 2], np.int64)].form == resultv2.form

    v2b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    resultv2 = v2b[np.array([0, 1, 2], np.int64)]
    assert to_list(resultv2) == [1.1, None, 3.3]
    assert v2b.to_typetracer()[np.array([0, 1, 2], np.int64)].form == resultv2.form


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
    resultv2 = v2a[np.array([0, 1, 4], np.int64)]
    assert to_list(resultv2) == [0.0, 1.0, None]
    assert v2a.to_typetracer()[np.array([0, 1, 4], np.int64)].form == resultv2.form

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
    resultv2 = v2b[np.array([0, 1, 4], np.int64)]
    assert to_list(resultv2) == [0.0, 1.0, None]
    assert v2b.to_typetracer()[np.array([0, 1, 4], np.int64)].form == resultv2.form

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
    resultv2 = v2c[np.array([0, 1, 4], np.int64)]
    assert to_list(resultv2) == [0.0, 1.0, None]
    assert v2c.to_typetracer()[np.array([0, 1, 4], np.int64)].form == resultv2.form

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
    resultv2 = v2d[np.array([0, 1, 4], np.int64)]
    assert to_list(resultv2) == [0.0, 1.0, None]
    assert v2d.to_typetracer()[np.array([0, 1, 4], np.int64)].form == resultv2.form


def test_UnmaskedArray_NumpyArray():
    v2a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )
    resultv2 = v2a[np.array([0, 1, 3], np.int64)]
    assert to_list(resultv2) == [0.0, 1.1, 3.3]
    assert v2a.to_typetracer()[np.array([0, 1, 3], np.int64)].form == resultv2.form


def test_UnionArray_NumpyArray():
    v2a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.from_iter([[1], [2], [3]], highlevel=False),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    resultv2 = v2a[np.array([0, 1, 3], np.int64)]
    assert to_list(resultv2) == [5.5, 4.4, [2]]
    assert v2a.to_typetracer()[np.array([0, 1, 3], np.int64)].form == resultv2.form


def test_RegularArray_RecordArray_NumpyArray():
    v2a = ak.contents.regulararray.RegularArray(
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
    resultv2 = v2a._carry(ak.index.Index(np.array([0], np.int64)), False)
    assert to_list(resultv2) == [[{"nest": 0.0}, {"nest": 1.1}, {"nest": 2.2}]]
    assert (
        v2a.to_typetracer()._carry(ak.index.Index(np.array([0], np.int64)), False).form
        == resultv2.form
    )

    v2b = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )
    resultv2 = v2b._carry(ak.index.Index(np.array([0], np.int64)), False)
    assert to_list(resultv2) == [[]]
    assert (
        v2b.to_typetracer()._carry(ak.index.Index(np.array([0], np.int64)), False).form
        == resultv2.form
    )


def test_ListArray_RecordArray_NumpyArray():
    v2a = ak.contents.listarray.ListArray(
        ak.index.Index(np.array([4, 100, 1], np.int64)),
        ak.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
                )
            ],
            ["nest"],
        ),
    )
    resultv2 = v2a[np.array([0, 1], np.int64)]
    assert to_list(resultv2) == [[{"nest": 1.1}, {"nest": 2.2}, {"nest": 3.3}], []]
    assert v2a.to_typetracer()[np.array([0, 1], np.int64)].form == resultv2.form


def test_ListOffsetArray_RecordArray_NumpyArray():
    v2a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6], np.int64)),
        ak.contents.recordarray.RecordArray(
            [ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])],
            ["nest"],
        ),
    )
    resultv2 = v2a[np.array([1, 2], np.int64)]
    assert to_list(resultv2) == [[], [{"nest": 4.4}, {"nest": 5.5}]]
    assert v2a.to_typetracer()[np.array([1, 2], np.int64)].form == resultv2.form


def test_IndexedArray_RecordArray_NumpyArray():
    v2a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    resultv2 = v2a[np.array([0, 1, 4], np.int64)]
    assert to_list(resultv2) == [{"nest": 3.3}, {"nest": 3.3}, {"nest": 5.5}]
    assert v2a.to_typetracer()[np.array([0, 1, 4], np.int64)].form == resultv2.form


def test_IndexedOptionArray_RecordArray_NumpyArray():
    v2a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak.contents.recordarray.RecordArray(
            [
                ak.contents.numpyarray.NumpyArray(
                    np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
                )
            ],
            ["nest"],
        ),
    )
    resultv2 = v2a[np.array([0, 1, 4], np.int64)]
    assert to_list(resultv2) == [{"nest": 3.3}, {"nest": 3.3}, None]
    assert v2a.to_typetracer()[np.array([0, 1, 4], np.int64)].form == resultv2.form


def test_ByteMaskedArray_RecordArray_NumpyArray():
    v2a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
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
    resultv2 = v2a._carry(ak.index.Index(np.array([0, 1, 4], np.int64)), False)
    assert to_list(resultv2) == [{"nest": 1.1}, None, {"nest": 5.5}]
    assert (
        v2a.to_typetracer()
        ._carry(ak.index.Index(np.array([0, 1, 4], np.int64)), False)
        .form
        == resultv2.form
    )

    v2b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
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
    resultv2 = v2b._carry(ak.index.Index(np.array([3, 1, 4], np.int64)), False)
    assert to_list(resultv2) == [None, None, {"nest": 5.5}]
    assert (
        v2b.to_typetracer()
        ._carry(ak.index.Index(np.array([3, 1, 4], np.int64)), False)
        .form
        == resultv2.form
    )


def test_BitMaskedArray_RecordArray_NumpyArray():
    v2a = ak.contents.bitmaskedarray.BitMaskedArray(
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
    resultv2 = v2a._carry(ak.index.Index(np.array([0, 1, 4], np.int64)), False)
    assert to_list(resultv2) == [{"nest": 0.0}, {"nest": 1.0}, None]
    assert (
        v2a.to_typetracer()
        ._carry(ak.index.Index(np.array([0, 1, 4], np.int64)), False)
        .form
        == resultv2.form
    )

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
    resultv2 = v2b._carry(ak.index.Index(np.array([1, 1, 4], np.int64)), False)
    assert to_list(resultv2) == [{"nest": 1.0}, {"nest": 1.0}, None]
    assert (
        v2b.to_typetracer()
        ._carry(ak.index.Index(np.array([1, 1, 4], np.int64)), False)
        .form
        == resultv2.form
    )

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
    resultv2 = v2c._carry(ak.index.Index(np.array([0, 1, 4], np.int64)), False)
    assert to_list(resultv2) == [{"nest": 0.0}, {"nest": 1.0}, None]
    assert (
        v2c.to_typetracer()
        ._carry(ak.index.Index(np.array([0, 1, 4], np.int64)), False)
        .form
        == resultv2.form
    )

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
    resultv2 = v2d._carry(ak.index.Index(np.array([0, 0, 0], np.int64)), False)
    assert to_list(resultv2) == [{"nest": 0.0}, {"nest": 0.0}, {"nest": 0.0}]
    assert (
        v2d.to_typetracer()
        ._carry(ak.index.Index(np.array([0, 0, 0], np.int64)), False)
        .form
        == resultv2.form
    )


def test_UnmaskedArray_RecordArray_NumpyArray():
    v2a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))],
            ["nest"],
        )
    )
    resultv2 = v2a._carry(ak.index.Index(np.array([0, 1, 1, 1, 1], np.int64)), False)
    assert to_list(resultv2) == [
        {"nest": 0.0},
        {"nest": 1.1},
        {"nest": 1.1},
        {"nest": 1.1},
        {"nest": 1.1},
    ]
    assert (
        v2a.to_typetracer()
        ._carry(ak.index.Index(np.array([0, 1, 1, 1, 1], np.int64)), False)
        .form
        == resultv2.form
    )


def test_UnionArray_RecordArray_NumpyArray():
    v2a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak.contents.recordarray.RecordArray(
                [ak.from_iter([[1], [2], [3]], highlevel=False)],
                ["nest"],
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
    resultv2 = v2a[np.array([0, 1, 1], np.int64)]
    assert to_list(resultv2) == [{"nest": 5.5}, {"nest": 4.4}, {"nest": 4.4}]
    assert v2a.to_typetracer()[np.array([0, 1, 1], np.int64)].form == resultv2.form


def test_RecordArray_NumpyArray_lazy():
    v2a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )
    resultv2 = v2a._carry(ak.index.Index(np.array([1, 2], np.int64)), True)
    assert to_list(resultv2) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}]
    assert (
        v2a.to_typetracer()
        ._carry(ak.index.Index(np.array([1, 2], np.int64)), True)
        .form
        == resultv2.form
    )

    v2b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )
    resultv2 = v2b._carry(ak.index.Index(np.array([0, 1, 2, 3, 4], np.int64)), True)
    assert to_list(resultv2) == [(0, 0.0), (1, 1.1), (2, 2.2), (3, 3.3), (4, 4.4)]
    assert (
        v2b.to_typetracer()
        ._carry(ak.index.Index(np.array([0, 1, 2, 3, 4], np.int64)), True)
        .form
        == resultv2.form
    )

    v2c = ak.contents.recordarray.RecordArray([], [], 10)
    resultv2 = v2c[np.array([0], np.int64)]
    assert to_list(resultv2) == [{}]
    assert v2c.to_typetracer()[np.array([0], np.int64)].form == resultv2.form

    v2d = ak.contents.recordarray.RecordArray([], None, 10)
    resultv2 = v2d[np.array([0], np.int64)]
    assert to_list(resultv2) == [()]
    assert v2d.to_typetracer()[np.array([0], np.int64)].form == resultv2.form


def test_reshaping():
    v2 = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )

    resultv2 = v2[ak.contents.NumpyArray(np.array([3, 6, 9, 2, 2, 1], np.int64))]
    assert to_list(resultv2) == [3.3, 6.6, 9.9, 2.2, 2.2, 1.1]
    assert (
        v2.to_typetracer()[
            ak.contents.NumpyArray(np.array([3, 6, 9, 2, 2, 1], np.int64))
        ].form
        == resultv2.form
    )

    resultv2 = v2[ak.contents.NumpyArray(np.array([[3, 6, 9], [2, 2, 1]], np.int64))]
    assert to_list(resultv2) == [[3.3, 6.6, 9.9], [2.2, 2.2, 1.1]]
    assert (
        v2.to_typetracer()[
            ak.contents.NumpyArray(np.array([[3, 6, 9], [2, 2, 1]], np.int64))
        ].form
        == resultv2.form
    )

    assert (
        str(
            ak.highlevel.Array(
                v2[ak.contents.NumpyArray(np.ones((2, 3), np.int64))]
            ).type
        )
        == "2 * 3 * float64"
    )

    assert (
        str(
            ak.highlevel.Array(
                v2[ak.contents.NumpyArray(np.ones((0, 3), np.int64))]
            ).type
        )
        == "0 * 3 * float64"
    )

    assert (
        str(
            ak.highlevel.Array(
                v2[ak.contents.NumpyArray(np.ones((2, 0, 3), np.int64))]
            ).type
        )
        == "2 * 0 * 3 * float64"
    )

    assert (
        str(
            ak.highlevel.Array(
                v2[ak.contents.NumpyArray(np.ones((1, 2, 0, 3), np.int64))]
            ).type
        )
        == "1 * 2 * 0 * 3 * float64"
    )
