# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np


def test_record():
    array = ak.Array(
        [
            {"x": 10},
            {"x": 11},
            {"x": 12},
        ]
    )

    assert not ak.is_tuple(array)
    assert not array.layout.istuple


def test_tuple():
    array = ak.Array(
        [
            (10,),
            (11,),
            (12,),
        ]
    )

    assert ak.is_tuple(array)
    assert array.layout.istuple


def test_numpy():
    array = ak.Array(ak.layout.NumpyArray(np.arange(10)))

    assert not ak.is_tuple(array)
    assert not array.layout.istuple


def test_list():
    tuple = ak.Array(
        ak.layout.ListArray64(
            ak.layout.Index64(np.array([0, 2], dtype=np.int64)),
            ak.layout.Index64(np.array([2, 4], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], None),
        )
    )

    assert ak.is_tuple(tuple)
    assert tuple.layout.istuple

    record = ak.Array(
        ak.layout.ListArray64(
            ak.layout.Index64(np.array([0, 2], dtype=np.int64)),
            ak.layout.Index64(np.array([2, 4], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], ["x"]),
        )
    )

    assert not ak.is_tuple(record)
    assert not record.layout.istuple


def test_listoffset():
    tuple = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([0, 2, 4], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], None),
        )
    )

    assert ak.is_tuple(tuple)
    assert tuple.layout.istuple

    record = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([0, 2, 4], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], ["x"]),
        )
    )

    assert not ak.is_tuple(record)
    assert not record.layout.istuple


def test_layouted():
    tuple = ak.Array(
        ak.layout.IndexedArray64(
            ak.layout.Index64(np.array([0, 1, 3], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], None),
        )
    )

    assert ak.is_tuple(tuple)
    assert tuple.layout.istuple

    record = ak.Array(
        ak.layout.IndexedArray64(
            ak.layout.Index64(np.array([0, 1, 3], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], ["x"]),
        )
    )

    assert not ak.is_tuple(record)
    assert not record.layout.istuple


def test_bytemasked():
    tuple = ak.Array(
        ak.layout.ByteMaskedArray(
            ak.layout.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], None),
            valid_when=True,
        )
    )

    assert ak.is_tuple(tuple)
    assert tuple.layout.istuple

    record = ak.Array(
        ak.layout.ByteMaskedArray(
            ak.layout.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], ["x"]),
            valid_when=True,
        )
    )

    assert not ak.is_tuple(record)
    assert not record.layout.istuple


def test_bitmasked():
    tuple = ak.Array(
        ak.layout.BitMaskedArray(
            ak.layout.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], None),
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert ak.is_tuple(tuple)
    assert tuple.layout.istuple

    record = ak.Array(
        ak.layout.BitMaskedArray(
            ak.layout.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], ["x"]),
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert not ak.is_tuple(record)
    assert not record.layout.istuple


def test_union():
    tuple = ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], None)

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [tuple, ak.layout.NumpyArray(np.arange(10))],
        )
    )

    assert not ak.is_tuple(array)
    assert not array.layout.istuple

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [tuple, tuple],
        )
    )

    assert ak.is_tuple(array)
    assert array.layout.istuple

    record = ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], ["x"])

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [record, ak.layout.NumpyArray(np.arange(10))],
        )
    )

    assert not ak.is_tuple(array)
    assert not array.layout.istuple

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [record, tuple],
        )
    )

    assert not ak.is_tuple(array)
    assert not array.layout.istuple

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [record, record],
        )
    )

    assert not ak.is_tuple(array)
    assert not array.layout.istuple
