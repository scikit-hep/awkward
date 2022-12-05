# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

tuple = ak.contents.RecordArray([ak.contents.NumpyArray(np.arange(10))], None)
tuple2 = ak.contents.RecordArray(
    [ak.from_iter([str(x) for x in range(10)], highlevel=False)], None
)
record = ak.contents.RecordArray([ak.contents.NumpyArray(np.arange(10))], ["x"])
record2 = ak.contents.RecordArray(
    [ak.from_iter([str(x) for x in range(10)], highlevel=False)], ["x"]
)


def test_record():
    array = ak.Array(record)

    assert not ak.is_tuple(array)


def test_tuple():
    array = ak.Array(tuple)

    assert ak.is_tuple(array)


def test_numpy():
    array = ak.Array(ak.contents.NumpyArray(np.arange(10)))

    assert not ak.is_tuple(array)


def test_list():
    array = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak.index.Index64(np.array([2, 4], dtype=np.int64)),
            tuple,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.contents.ListArray(
            ak.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak.index.Index64(np.array([2, 4], dtype=np.int64)),
            record,
        )
    )

    assert not ak.is_tuple(array)


def test_listoffset():
    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2, 4], dtype=np.int64)),
            tuple,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2, 4], dtype=np.int64)), record
        )
    )

    assert not ak.is_tuple(array)


def test_indexed():
    array = ak.Array(
        ak.contents.IndexedArray(
            ak.index.Index64(np.array([0, 1, 3], dtype=np.int64)), tuple
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.contents.IndexedArray(
            ak.index.Index64(np.array([0, 1, 3], dtype=np.int64)), record
        )
    )

    assert not ak.is_tuple(array)


def test_regular():
    array = ak.Array(ak.contents.RegularArray(tuple, 5))

    assert ak.is_tuple(array)

    array = ak.Array(ak.contents.RegularArray(record, 5))

    assert not ak.is_tuple(array)


def test_bytemasked():
    array = ak.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            tuple,
            valid_when=True,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.contents.ByteMaskedArray(
            ak.index.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            record,
            valid_when=True,
        )
    )

    assert not ak.is_tuple(array)


def test_bitmasked():
    array = ak.Array(
        ak.contents.BitMaskedArray(
            ak.index.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            tuple,
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.contents.BitMaskedArray(
            ak.index.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            record,
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert not ak.is_tuple(array)


def test_union():
    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 1, 1]),
            ak.index.Index64([0, 1, 0, 1]),
            [tuple, ak.contents.NumpyArray(np.arange(10))],
        )
    )

    assert not ak.is_tuple(array)

    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 1, 1]),
            ak.index.Index64([0, 1, 0, 1]),
            [tuple, tuple2],
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 1, 1]),
            ak.index.Index64([0, 1, 0, 1]),
            [record, ak.contents.NumpyArray(np.arange(10))],
        )
    )

    assert not ak.is_tuple(array)

    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 1, 1]),
            ak.index.Index64([0, 1, 0, 1]),
            [record, tuple],
        )
    )

    assert not ak.is_tuple(array)

    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8([0, 0, 1, 1]),
            ak.index.Index64([0, 1, 0, 1]),
            [record, record2],
        )
    )

    assert not ak.is_tuple(array)
