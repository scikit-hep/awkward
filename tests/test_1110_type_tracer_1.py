# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._typetracer import UnknownLength

typetracer = ak._typetracer.TypeTracer.instance()


def test_getitem_at():
    concrete = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(2, 3, 5) * 0.1)
    abstract = concrete.to_typetracer()

    assert concrete.shape == (2, 3, 5)
    assert abstract.shape[1:] == (3, 5)
    assert abstract[0].shape[1:] == (5,)
    assert abstract[0][0].shape[1:] == ()

    assert abstract.form == concrete.form
    assert abstract.form.type == concrete.form.type

    assert abstract[0].form == concrete[0].form
    assert abstract[0].form.type == concrete[0].form.type


def test_EmptyArray():
    a = ak.contents.emptyarray.EmptyArray()
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form


def test_NumpyArray():
    a = ak.contents.numpyarray.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

    b = ak.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength
    assert b.to_typetracer(forget_length=True).data.shape[1:] == (3, 5)


def test_RegularArray_NumpyArray():
    # 6.6 is inaccessible
    a = ak.contents.regulararray.RegularArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
        ),
        3,
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

    b = ak.contents.regulararray.RegularArray(
        ak.contents.emptyarray.EmptyArray(), 0, zeros_length=10
    )
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


def test_ListOffsetArray_NumpyArray():
    # 6.6 and 7.7 are inaccessible
    a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6])),
        ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


def test_RecordArray_NumpyArray():
    # 5.5 is inaccessible
    a = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4])),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        ["x", "y"],
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

    # 5.5 is inaccessible
    b = ak.contents.recordarray.RecordArray(
        [
            ak.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4])),
            ak.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
        None,
    )
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength

    c = ak.contents.recordarray.RecordArray([], [], 10)
    assert c.to_typetracer().form == c.to_typetracer(forget_length=True).form
    assert c.to_typetracer(forget_length=True).length is UnknownLength

    d = ak.contents.recordarray.RecordArray([], None, 10)
    assert d.to_typetracer().form == d.to_typetracer(forget_length=True).form
    assert d.to_typetracer(forget_length=True).length is UnknownLength


def test_IndexedArray_NumpyArray():
    # 4.4 is inaccessible; 3.3 and 5.5 appear twice
    a = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np.array([2, 2, 0, 1, 4, 5, 4])),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


def test_IndexedOptionArray_NumpyArray():
    # 1.1 and 4.4 are inaccessible; 3.3 appears twice
    a = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np.array([2, 2, -1, 1, -1, 5, 4])),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


def test_ByteMaskedArray_NumpyArray():
    # 2.2, 4.4, and 6.6 are inaccessible
    a = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([1, 0, 1, 0, 1], dtype=np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

    # 2.2, 4.4, and 6.6 are inaccessible
    b = ak.contents.bytemaskedarray.ByteMaskedArray(
        ak.index.Index(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
        ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

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
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength

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
    assert c.to_typetracer().form == c.to_typetracer(forget_length=True).form
    assert c.to_typetracer(forget_length=True).length is UnknownLength

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
    assert d.to_typetracer().form == d.to_typetracer(forget_length=True).form
    assert d.to_typetracer(forget_length=True).length is UnknownLength


def test_UnmaskedArray_NumpyArray():
    a = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(
            np.array([0.0, 1.1, 2.2, 3.3], dtype=np.float64)
        )
    )
    assert a.to_typetracer().form == a.form
    assert a.to_typetracer().form.type == a.form.type
    assert len(a) == 4
    assert a[2] == 2.2
    assert a[-2] == 2.2
    assert type(a[2]) is np.float64
    with pytest.raises(IndexError):
        a[4]
    with pytest.raises(IndexError):
        a[-5]
    assert isinstance(a[2:], ak.contents.unmaskedarray.UnmaskedArray)
    assert a[2:][0] == 2.2
    assert len(a[2:]) == 2
    with pytest.raises(IndexError):
        a["bad"]
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


def test_UnionArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.from_iter(["1", "2", "3"], highlevel=False),
            ak.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

    b = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.emptyarray.EmptyArray()], ["nest"]
        ),
        0,
        zeros_length=10,
    )
    assert b.to_typetracer().form == b.form
    assert b.to_typetracer().form.type == b.form.type
    assert len(b["nest"]) == 10
    assert b.to_typetracer()["nest"].form == b["nest"].form
    assert isinstance(b["nest"][5], ak.contents.emptyarray.EmptyArray)
    assert b.to_typetracer()["nest"][5].form == b["nest"][5].form
    assert len(b["nest"][5]) == 0
    assert isinstance(b["nest"][7:], ak.contents.regulararray.RegularArray)
    assert b.to_typetracer()["nest"][7:].form == b["nest"][7:].form
    assert len(b["nest"][7:]) == 3
    assert len(b["nest"][7:100]) == 3
    with pytest.raises(IndexError):
        b["nest"]["bad"]
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


def test_ListOffsetArray_RecordArray_NumpyArray():
    # 6.6 and 7.7 are inaccessible
    a = ak.contents.listoffsetarray.ListOffsetArray(
        ak.index.Index(np.array([1, 4, 4, 6])),
        ak.contents.recordarray.RecordArray(
            [ak.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7])],
            ["nest"],
        ),
    )
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

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
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength

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
    assert b.to_typetracer().form == b.to_typetracer(forget_length=True).form
    assert b.to_typetracer(forget_length=True).length is UnknownLength

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
    assert c.to_typetracer().form == c.to_typetracer(forget_length=True).form
    assert c.to_typetracer(forget_length=True).length is UnknownLength

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
    assert d.to_typetracer().form == d.to_typetracer(forget_length=True).form
    assert d.to_typetracer(forget_length=True).length is UnknownLength


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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength


def test_UnionArray_RecordArray_NumpyArray():
    # 100 is inaccessible in index
    # 1.1 is inaccessible in contents[1]
    a = ak.contents.unionarray.UnionArray(
        ak.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)),
        ak.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100])),
        [
            ak.contents.recordarray.RecordArray(
                [ak.from_iter(["1", "2", "3"], highlevel=False)], ["nest"]
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
    assert a.to_typetracer().form == a.to_typetracer(forget_length=True).form
    assert a.to_typetracer(forget_length=True).length is UnknownLength
