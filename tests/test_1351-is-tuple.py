# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np

tuple = ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], None)
record = ak.layout.RecordArray([ak.layout.NumpyArray(np.arange(10))], ["x"])


def test_record():
    array = ak.Array(record)

    assert not ak.is_tuple(array)


def test_tuple():
    array = ak.Array(tuple)

    assert ak.is_tuple(array)


def test_numpy():
    array = ak.Array(ak.layout.NumpyArray(np.arange(10)))

    assert not ak.is_tuple(array)


def test_list():
    array = ak.Array(
        ak.layout.ListArray64(
            ak.layout.Index64(np.array([0, 2], dtype=np.int64)),
            ak.layout.Index64(np.array([2, 4], dtype=np.int64)),
            tuple,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.layout.ListArray64(
            ak.layout.Index64(np.array([0, 2], dtype=np.int64)),
            ak.layout.Index64(np.array([2, 4], dtype=np.int64)),
            record,
        )
    )

    assert not ak.is_tuple(array)


def test_listoffset():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([0, 2, 4], dtype=np.int64)),
            tuple,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64(np.array([0, 2, 4], dtype=np.int64)), record
        )
    )

    assert not ak.is_tuple(array)


def test_indexed():
    array = ak.Array(
        ak.layout.IndexedArray64(
            ak.layout.Index64(np.array([0, 1, 3], dtype=np.int64)), tuple
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.layout.IndexedArray64(
            ak.layout.Index64(np.array([0, 1, 3], dtype=np.int64)), record
        )
    )

    assert not ak.is_tuple(array)


def test_regular():
    array = ak.Array(ak.layout.RegularArray(tuple, 5))

    assert ak.is_tuple(array)

    array = ak.Array(ak.layout.RegularArray(record, 5))

    assert not ak.is_tuple(array)


def test_bytemasked():
    array = ak.Array(
        ak.layout.ByteMaskedArray(
            ak.layout.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            tuple,
            valid_when=True,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.layout.ByteMaskedArray(
            ak.layout.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            record,
            valid_when=True,
        )
    )

    assert not ak.is_tuple(array)


def test_bitmasked():
    array = ak.Array(
        ak.layout.BitMaskedArray(
            ak.layout.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            tuple,
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.layout.BitMaskedArray(
            ak.layout.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            record,
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert not ak.is_tuple(array)


def test_union():
    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [tuple, ak.layout.NumpyArray(np.arange(10))],
        )
    )

    assert not ak.is_tuple(array)

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [tuple, tuple],
        )
    )

    assert ak.is_tuple(array)

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [record, ak.layout.NumpyArray(np.arange(10))],
        )
    )

    assert not ak.is_tuple(array)

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [record, tuple],
        )
    )

    assert not ak.is_tuple(array)

    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8([0, 0, 1, 1]),
            ak.layout.Index64([0, 1, 0, 1]),
            [record, record],
        )
    )

    assert not ak.is_tuple(array)
