# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak


def test():
    np_data = np.random.random(size=(4, 100 * 1024 * 1024 // 8 // 4))
    array = ak.operations.from_numpy(np_data, regulararray=False)

    assert np_data.nbytes == array.nbytes


def test_NumpyArray_nbytes():
    np_data = np.random.random(size=(4, 100 * 1024 * 1024 // 8 // 4))
    array = ak.contents.numpyarray.NumpyArray(np_data)
    assert array.nbytes == np_data.nbytes


def test_ByteMaskedArray_nbytes():
    content = ak.operations.from_iter(
        [
            [[1.1, 0.0, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 9.9, 8.8, 7.7]],
            [[], [12.2, 11.1, 10.0]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert array.nbytes == 221


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
    array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(np.packbits(np_index)),
        ak.contents.numpyarray.NumpyArray(np_array),
        valid_when=True,
        length=13,
        lsb_order=False,
    )

    assert np_array.nbytes == np_array.dtype.itemsize * len(np_array)
    assert np_index.nbytes == np_index.dtype.itemsize * len(np_index)
    assert np.packbits(np_index).nbytes == 2
    assert array.nbytes == np_array.nbytes + np.packbits(np_index).nbytes

    array = ak.contents.bitmaskedarray.BitMaskedArray(
        ak.index.Index(np.packbits(np_index)),
        ak.contents.numpyarray.NumpyArray(
            np_array,
        ),
        valid_when=True,
        length=13,
        lsb_order=False,
    )
    assert array.nbytes == np_array.nbytes + np.packbits(np_index).nbytes


def test_EmptyArray_nbytes():
    array = ak.contents.emptyarray.EmptyArray()
    assert array.nbytes == 0


def test_IndexedArray_nbytes():
    np_index = np.array([2, 2, 0, 1, 4, 5, 4])
    np_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    array = ak.contents.indexedarray.IndexedArray(
        ak.index.Index(np_index),
        ak.contents.numpyarray.NumpyArray(np_content),
    )
    assert array.nbytes == np_index.nbytes + np_content.nbytes


def test_IndexedOptionArray_nbytes():
    np_index = np.array([2, 2, -1, 1, -1, 5, 4])
    np_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    array = ak.contents.indexedoptionarray.IndexedOptionArray(
        ak.index.Index(np_index),
        ak.contents.numpyarray.NumpyArray(np_content),
    )
    assert array.nbytes == np_index.nbytes + np_content.nbytes


def test_ListArray_nbytes():
    np_starts = np.array([4, 100, 1])
    np_stops = np.array([7, 100, 3, 200])
    np_content = np.array([6.6, 4.4, 5.5, 7.7, 3.3, 2.2, 1.1, 8.8])
    array = ak.contents.listarray.ListArray(
        ak.index.Index(np_starts),
        ak.index.Index(np_stops),
        ak.contents.numpyarray.NumpyArray(np_content),
    )
    assert array.nbytes == np_starts.nbytes + np_stops.nbytes + np_content.nbytes


def test_ListOffsetArray_nbytes():
    np_offsets = np.array([7, 10, 10, 200])
    np_content = np.array([6.6, 4.4, 5.5, 7.7, 3.3, 2.2, 1.1, 8.8])
    array = ak.contents.ListOffsetArray(
        ak.index.Index(np_offsets),
        ak.contents.numpyarray.NumpyArray(np_content),
    )
    assert array.nbytes == np_offsets.nbytes + np_content.nbytes


def test_RecordArray_nbytes():
    np_content = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    array = ak.contents.recordarray.RecordArray(
        [ak.contents.numpyarray.NumpyArray(np_content)],
        ["nest"],
    )
    assert array.nbytes == np_content.nbytes


def test_RegularArray_nbytes():
    np_content = np.array([0.0, 1.1, 2.2, 33.33, 4.4, 5.5, -6.6])
    array = ak.contents.regulararray.RegularArray(
        ak.contents.recordarray.RecordArray(
            [ak.contents.numpyarray.NumpyArray(np_content)],
            ["nest"],
        ),
        3,
    )
    assert array.nbytes == np_content.nbytes


def test_UnionArray_nbytes():
    np_tags = np.array([1, 1, 0, 0, 1, 0, 1], dtype=np.int8)
    np_index = np.array([4, 3, 0, 1, 2, 2, 4, 100])
    np_content1 = np.array([1, 2, 3])
    np_content2 = np.array([[1.1], [2.2], [3.3], [4.4], [5.5]])
    array = ak.contents.unionarray.UnionArray(
        ak.index.Index(np_tags),
        ak.index.Index(np_index),
        [
            ak.contents.numpyarray.NumpyArray(np_content1),
            ak.contents.numpyarray.NumpyArray(np_content2),
        ],
    )
    assert (
        array.nbytes
        == np_tags.nbytes + np_index.nbytes + np_content1.nbytes + np_content2.nbytes
    )


def test_UnmaskedArray_nbytes():
    np_content = np.array([0.0, 2.2, 1.1, 3.3], dtype=np.float64)
    array = ak.contents.unmaskedarray.UnmaskedArray(
        ak.contents.numpyarray.NumpyArray(np_content)
    )
    assert array.nbytes == np_content.nbytes


def test_highlevel():
    ak_Array = ak.highlevel.Array
    array = ak_Array([0.0, 1.1, 2.2, 3.3, 4.4])
    assert array.nbytes == array.layout.nbytes
