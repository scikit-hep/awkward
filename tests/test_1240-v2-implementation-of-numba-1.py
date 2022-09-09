# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys
import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

ak_numba = pytest.importorskip("awkward._v2.numba")
ak_numba_arrayview = pytest.importorskip("awkward._v2._connect.numba.arrayview")
ak_numba_layout = pytest.importorskip("awkward._v2._connect.numba.layout")

ak_numba.register_and_check()


def roundtrip(layout):
    assert isinstance(layout, ak._v2.contents.Content)

    lookup = ak._v2._lookup.Lookup(layout)
    assert isinstance(lookup, ak._v2._lookup.Lookup)

    numbatype = ak_numba_arrayview.tonumbatype(layout.form)
    assert isinstance(numbatype, ak_numba_layout.ContentType)

    layout2 = numbatype.tolayout(lookup, 0, ())
    assert isinstance(layout2, ak._v2.contents.Content)

    assert layout.to_list() == layout2.to_list()
    assert layout.form.type == layout2.form.type


@numba.njit
def swallow(array):
    pass


@numba.njit
def passthrough(array):
    return array


@numba.njit
def passthrough2(array):
    return array, array


@numba.njit
def digest(array):
    return array[0]


@numba.njit
def digest2(array):
    tmp = array[0]
    return tmp, tmp, array[0]


def buffers(layout):
    if isinstance(layout, ak._v2.contents.NumpyArray):
        yield layout.data
    for attr in dir(layout):
        obj = getattr(layout, attr)
        if isinstance(obj, ak._v2.index.Index):
            yield obj.data
        elif attr == "content":
            yield from buffers(obj)
        elif attr == "contents":
            for x in obj:
                yield from buffers(x)


def memoryleak(array, function):
    counts = None
    for _ in range(10):
        function(array)
        this_counts = [sys.getrefcount(x) for x in buffers(array.layout)]
        if counts is None:
            counts = this_counts
        else:
            assert counts == this_counts


def test_EmptyArray():
    v2a = ak._v2.contents.emptyarray.EmptyArray().toNumpyArray(np.dtype(np.float64))
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)


def test_NumpyArray():
    v2a = ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

    v2b = ak._v2.contents.numpyarray.NumpyArray(
        np.arange(2 * 3 * 5, dtype=np.int64).reshape(2, 3, 5)
    )
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_RegularArray_NumpyArray():
    v2a = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])),
        3,
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

    v2b = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.emptyarray.EmptyArray().toNumpyArray(np.dtype(np.float64)),
        0,
        zeros_length=10,
    )
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_ListArray_NumpyArray():
    v2a = ak._v2.contents.listarray.ListArray(
        ak._v2.index.Index(np.array([4, 100, 1], np.int64)),
        ak._v2.index.Index(np.array([7, 100, 3, 200], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(
            np.array([6.6, 4.4, 5.5, 7.7, 1.1, 2.2, 3.3, 8.8])
        ),
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_ListOffsetArray_NumpyArray():
    v2a = ak._v2.contents.listoffsetarray.ListOffsetArray(
        ak._v2.index.Index(np.array([1, 4, 4, 6, 7], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray([6.6, 1.1, 2.2, 3.3, 4.4, 5.5, 7.7]),
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

    v2b = ak._v2.contents.recordarray.RecordArray(
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([0, 1, 2, 3, 4], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(
                np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5])
            ),
        ],
        None,
    )
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

    v2c = ak._v2.contents.recordarray.RecordArray([], [], 10)
    roundtrip(v2c)
    array = ak._v2.highlevel.Array(v2c)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

    v2d = ak._v2.contents.recordarray.RecordArray([], None, 10)
    roundtrip(v2d)
    array = ak._v2.highlevel.Array(v2d)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_IndexedArray_NumpyArray():
    v2a = ak._v2.contents.indexedarray.IndexedArray(
        ak._v2.index.Index(np.array([2, 2, 0, 1, 4, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_IndexedOptionArray_NumpyArray():
    v2a = ak._v2.contents.indexedoptionarray.IndexedOptionArray(
        ak._v2.index.Index(np.array([2, 2, -1, 1, -1, 5, 4], np.int64)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_ByteMaskedArray_NumpyArray():
    v2a = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([1, 0, 1, 0, 1], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=True,
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

    v2b = ak._v2.contents.bytemaskedarray.ByteMaskedArray(
        ak._v2.index.Index(np.array([0, 1, 0, 1, 0], np.int8)),
        ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])),
        valid_when=False,
    )
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

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
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

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
    roundtrip(v2c)
    array = ak._v2.highlevel.Array(v2c)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

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
    roundtrip(v2d)
    array = ak._v2.highlevel.Array(v2c)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_UnmaskedArray_NumpyArray():
    v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_UnionArray_NumpyArray():
    v2a = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np.array([1, 1, 0, 0, 1, 0, 1], np.int8)),
        ak._v2.index.Index(np.array([4, 3, 0, 1, 2, 2, 4, 100], np.int64)),
        [
            ak._v2.contents.numpyarray.NumpyArray(np.array([1, 2, 3], np.int64)),
            ak._v2.contents.numpyarray.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

    v2b = ak._v2.contents.regulararray.RegularArray(
        ak._v2.contents.recordarray.RecordArray(
            [
                ak._v2.contents.emptyarray.EmptyArray().toNumpyArray(
                    np.dtype(np.float64)
                )
            ],
            ["nest"],
        ),
        0,
        zeros_length=10,
    )
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

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
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

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
    roundtrip(v2b)
    array = ak._v2.highlevel.Array(v2b)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

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
    roundtrip(v2c)
    array = ak._v2.highlevel.Array(v2c)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)

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
    roundtrip(v2d)
    array = ak._v2.highlevel.Array(v2d)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


def test_UnmaskedArray_RecordArray_NumpyArray():
    v2a = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.recordarray.RecordArray(
            [ak._v2.contents.numpyarray.NumpyArray(np.array([0.0, 1.1, 2.2, 3.3]))],
            ["nest"],
        )
    )
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
    memoryleak(array, digest)
    memoryleak(array, digest2)


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
    roundtrip(v2a)
    array = ak._v2.highlevel.Array(v2a)
    memoryleak(array, swallow)
    memoryleak(array, passthrough)
    memoryleak(array, passthrough2)
