# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import pytest  # noqa: F401
import awkward as ak  # noqa: F401
import numpy as np

tuple = ak._v2.contents.RecordArray([ak._v2.contents.NumpyArray(np.arange(10))], None)
record = ak._v2.contents.RecordArray([ak._v2.contents.NumpyArray(np.arange(10))], ["x"])


def test_record():
    array = ak._v2.Array(record)

    assert not ak._v2.is_tuple(array)


def test_tuple():
    array = ak._v2.Array(tuple)

    assert ak._v2.is_tuple(array)


def test_numpy():
    array = ak._v2.Array(ak._v2.contents.NumpyArray(np.arange(10)))

    assert not ak._v2.is_tuple(array)


def test_list():
    array = ak._v2.Array(
        ak._v2.contents.ListArray(
            ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak._v2.index.Index64(np.array([2, 4], dtype=np.int64)),
            tuple,
        )
    )

    assert ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.ListArray(
            ak._v2.index.Index64(np.array([0, 2], dtype=np.int64)),
            ak._v2.index.Index64(np.array([2, 4], dtype=np.int64)),
            record,
        )
    )

    assert not ak._v2.is_tuple(array)


def test_listoffset():
    array = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2, 4], dtype=np.int64)),
            tuple,
        )
    )

    assert ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(np.array([0, 2, 4], dtype=np.int64)), record
        )
    )

    assert not ak._v2.is_tuple(array)


def test_indexed():
    array = ak._v2.Array(
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index64(np.array([0, 1, 3], dtype=np.int64)), tuple
        )
    )

    assert ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.IndexedArray(
            ak._v2.index.Index64(np.array([0, 1, 3], dtype=np.int64)), record
        )
    )

    assert not ak._v2.is_tuple(array)


def test_regular():
    array = ak._v2.Array(ak._v2.contents.RegularArray(tuple, 5))

    assert ak._v2.is_tuple(array)

    array = ak._v2.Array(ak._v2.contents.RegularArray(record, 5))

    assert not ak._v2.is_tuple(array)


def test_bytemasked():
    array = ak._v2.Array(
        ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            tuple,
            valid_when=True,
        )
    )

    assert ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index8(np.array([0, 1, 0, 1], dtype=np.int64)),
            record,
            valid_when=True,
        )
    )

    assert not ak._v2.is_tuple(array)


def test_bitmasked():
    array = ak._v2.Array(
        ak._v2.contents.BitMaskedArray(
            ak._v2.index.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            tuple,
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.BitMaskedArray(
            ak._v2.index.IndexU8(np.array([0, 1, 0, 1], dtype=np.int64)),
            record,
            valid_when=True,
            length=4,
            lsb_order=True,
        )
    )

    assert not ak._v2.is_tuple(array)


def test_union():
    array = ak._v2.Array(
        ak._v2.contents.UnionArray(
            ak._v2.index.Index8([0, 0, 1, 1]),
            ak._v2.index.Index64([0, 1, 0, 1]),
            [tuple, ak._v2.contents.NumpyArray(np.arange(10))],
        )
    )

    assert not ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.UnionArray(
            ak._v2.index.Index8([0, 0, 1, 1]),
            ak._v2.index.Index64([0, 1, 0, 1]),
            [tuple, tuple],
        )
    )

    assert ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.UnionArray(
            ak._v2.index.Index8([0, 0, 1, 1]),
            ak._v2.index.Index64([0, 1, 0, 1]),
            [record, ak._v2.contents.NumpyArray(np.arange(10))],
        )
    )

    assert not ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.UnionArray(
            ak._v2.index.Index8([0, 0, 1, 1]),
            ak._v2.index.Index64([0, 1, 0, 1]),
            [record, tuple],
        )
    )

    assert not ak._v2.is_tuple(array)

    array = ak._v2.Array(
        ak._v2.contents.UnionArray(
            ak._v2.index.Index8([0, 0, 1, 1]),
            ak._v2.index.Index64([0, 1, 0, 1]),
            [record, record],
        )
    )

    assert not ak._v2.is_tuple(array)
