# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test():
    np_data = np.random.random(size=(4, 100 * 1024 * 1024 // 8 // 4))
    array = ak.from_numpy(np_data, regulararray=False)
    array = v1_to_v2(array.layout)

    assert np_data.nbytes == array.nbytes


def test_NumpyArray_nbytes():
    np_data = np.random.random(size=(4, 100 * 1024 * 1024 // 8 // 4))

    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )

    assert identifier._nbytes_part() == np.dtype(np.int64).itemsize * 5 * 10

    array = ak._v2.contents.numpyarray.NumpyArray(np_data, identifier)
    assert array.nbytes == np_data.nbytes + np.dtype(np.int64).itemsize * 5 * 10


def test_ByteMaskedArray_nbytes():
    content = ak.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    v1_array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)
    v2_array = v1_to_v2(v1_array)

    assert v2_array.nbytes == v1_array.nbytes


def test_BitMaskedArray_nbytes():
    np_array = np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    )
    np_index = np.array(
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
    array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(np.packbits(np_index)),
        ak._v2.contents.numpyarray.NumpyArray(np_array),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert np_array.nbytes == np_array.dtype.itemsize * len(np_array)
    assert np_index.nbytes == np_index.dtype.itemsize * len(np_index)
    assert np.packbits(np_index).nbytes == 2
    assert array.nbytes == np_array.nbytes + np.packbits(np_index).nbytes

    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    array = ak._v2.contents.bitmaskedarray.BitMaskedArray(
        ak._v2.index.Index(np.packbits(np_index)),
        ak._v2.contents.numpyarray.NumpyArray(
            np_array,
            identifier,
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
        identifier=identifier,
    )
    assert (
        array.nbytes
        == np_array.nbytes
        + np.packbits(np_index).nbytes
        + 2 * np.dtype(np.int64).itemsize * 5 * 10
    )


def test_EmptyArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    array = ak._v2.contents.emptyarray.EmptyArray(
        identifier=identifier,
    )
    assert array.nbytes == np.dtype(np.int64).itemsize * 5 * 10


def test_IndexedArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_index = np.array([2, 2, 0, 1, 4, 5, 4])
    np_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    array = ak._v2.contents.indexedarray.IndexedArray(  # noqa: F841
        ak._v2.index.Index(np_index),
        ak._v2.contents.numpyarray.NumpyArray(np_content),
        identifier=identifier,
    )
    assert (
        array.nbytes
        == np_index.nbytes + np_content.nbytes + np.dtype(np.int64).itemsize * 5 * 10
    )


def test_IndexedOptionArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_index = np.array([2, 2, -1, 1, -1, 5, 4])
    np_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    array = ak._v2.contents.indexedoptionarray.IndexedOptionArray(  # noqa: F841
        ak._v2.index.Index(np_index),
        ak._v2.contents.numpyarray.NumpyArray(np_content),
        identifier=identifier,
    )
    assert (
        array.nbytes
        == np_index.nbytes + np_content.nbytes + np.dtype(np.int64).itemsize * 5 * 10
    )


def test_ListArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_starts = np.array([4, 100, 1])
    np_stops = np.array([7, 100, 3, 200])
    np_content = np.array([6.6, 4.4, 5.5, 7.7, 3.3, 2.2, 1.1, 8.8])
    array = ak._v2.contents.listarray.ListArray(  # noqa: F841
        ak._v2.index.Index(np_starts),
        ak._v2.index.Index(np_stops),
        ak._v2.contents.numpyarray.NumpyArray(np_content),
        identifier=identifier,
    )
    assert (
        array.nbytes
        == np_starts.nbytes
        + np_stops.nbytes
        + np_content.nbytes
        + np.dtype(np.int64).itemsize * 5 * 10
    )


def test_ListOffsetArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_offsets = np.array([7, 10, 10, 200])
    np_content = np.array([6.6, 4.4, 5.5, 7.7, 3.3, 2.2, 1.1, 8.8])
    array = ak._v2.contents.ListOffsetArray(
        ak._v2.index.Index(np_offsets),
        ak._v2.contents.numpyarray.NumpyArray(np_content),
        identifier=identifier,
    )
    assert (
        array.nbytes
        == np_offsets.nbytes + np_content.nbytes + np.dtype(np.int64).itemsize * 5 * 10
    )


def test_RecordArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    array = ak._v2.contents.recordarray.RecordArray(
        [ak._v2.contents.numpyarray.NumpyArray(np_content)],
        ["nest"],
        identifier=identifier,
    )
    assert array.nbytes == np_content.nbytes + np.dtype(np.int64).itemsize * 5 * 10


def test_RegularArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_content = np.array([0.0, 1.1, 2.2, 33.33, 4.4, 5.5, -6.6])
    array = ak._v2.contents.regulararray.RegularArray(  # noqa: F841
        ak._v2.contents.recordarray.RecordArray(
            [ak._v2.contents.numpyarray.NumpyArray(np_content)],
            ["nest"],
        ),
        3,
        identifier=identifier,
    )
    assert array.nbytes == np_content.nbytes + np.dtype(np.int64).itemsize * 5 * 10


def test_UnionArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_tags = np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)
    np_index = np.array([4, 3, 0, 1, 2, 2, 4, 100])
    np_content1 = np.array([1, 2, 3])
    np_content2 = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    array = ak._v2.contents.unionarray.UnionArray(
        ak._v2.index.Index(np_tags),
        ak._v2.index.Index(np_index),
        [
            ak._v2.contents.numpyarray.NumpyArray(np_content1),
            ak._v2.contents.numpyarray.NumpyArray(np_content2),
        ],
        identifier=identifier,
    )
    assert (
        array.nbytes
        == np_tags.nbytes
        + np_index.nbytes
        + np_content1.nbytes
        + np_content2.nbytes
        + np.dtype(np.int64).itemsize * 5 * 10
    )


def test_UnmaskedArray_nbytes():
    identifier = ak._v2.identifier.Identifier.zeros(
        123, {1: "one", 2: "two"}, 5, 10, np, np.int64
    )
    np_content = np.array([0.0, 2.2, 1.1, 3.3], dtype=np.float64)
    array = ak._v2.contents.unmaskedarray.UnmaskedArray(
        ak._v2.contents.numpyarray.NumpyArray(np_content),
        identifier=identifier,
    )
    assert array.nbytes == np_content.nbytes + np.dtype(np.int64).itemsize * 5 * 10


def test_highlevel():
    ak_Array = ak._v2.highlevel.Array
    array = ak_Array([0.0, 1.1, 2.2, 3.3, 4.4])
    assert array.nbytes == array.layout.nbytes
