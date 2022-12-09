# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_numpy_array():
    matrix = np.arange(64).reshape(8, -1)
    layout = ak.contents.NumpyArray(matrix[:, 0])
    assert not layout.is_contiguous

    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    assert packed.is_contiguous


def test_empty_array():
    layout = ak.contents.EmptyArray()
    assert ak.operations.to_packed(layout, highlevel=False) is layout


def test_indexed_option_array():
    index = ak.index.Index64(np.r_[0, -1, 2, -1, 4])
    content = ak.contents.NumpyArray(np.arange(8))
    layout = ak.contents.IndexedOptionArray(index, content)

    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(layout) == to_list(packed)
    assert isinstance(packed, ak.contents.IndexedOptionArray)
    assert np.asarray(packed.index).tolist() == [0, -1, 1, -1, 2]
    assert len(packed.content) == 3


def test_indexed_array():
    index = ak.index.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak.contents.NumpyArray(np.arange(10))
    layout = ak.contents.IndexedArray(index, content)

    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)

    assert isinstance(packed, ak.contents.NumpyArray)
    assert len(packed) == len(index)


def test_list_array():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
    layout = ak.contents.ListArray(starts, stops, content)

    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    assert isinstance(packed, ak.contents.ListOffsetArray)
    assert packed.offsets[0] == 0


def test_list_offset_array():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    layout = ak.contents.ListOffsetArray(offsets, content)

    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    assert isinstance(packed, ak.contents.ListOffsetArray)
    assert packed.offsets[0] == 0
    assert len(packed.content) == packed.offsets[-1]


def test_unmasked_array():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    layout = ak.contents.UnmaskedArray(content)
    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)


def test_union_array():
    a = ak.contents.NumpyArray(np.arange(4))
    b = ak.contents.NumpyArray(np.arange(4) + 4)
    c = ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(12)), 3)
    layout = ak.contents.UnionArray.simplified(
        ak.index.Index8([1, 1, 2, 2, 0, 0]),
        ak.index.Index64([0, 1, 0, 1, 0, 1]),
        [a, b, c],
    )
    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    # Check that it merges like contents
    assert len(packed.contents) == 2
    index_0 = np.asarray(packed.index)[np.asarray(packed.tags) == 0]
    assert index_0.tolist() == [0, 1, 2, 3]


def test_record_array():
    a = ak.contents.NumpyArray(np.arange(10))
    b = ak.contents.NumpyArray(np.arange(10) * 2 + 4)
    layout = ak.contents.RecordArray([a, b], None, 5)
    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    assert len(packed.contents[0]) == 5
    assert len(packed.contents[1]) == 5


def test_regular_array():
    content = ak.contents.NumpyArray(np.arange(10))
    layout = ak.contents.RegularArray(content, 3)
    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    assert len(packed.content) == 9
    assert packed.size == layout.size


def test_bit_masked_aray():
    mask = ak.index.IndexU8(np.array([0b10101010]))
    content = ak.contents.NumpyArray(np.arange(16))
    layout = ak.contents.BitMaskedArray(mask, content, False, 8, False)
    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    assert len(packed.content) == 8


def test_byte_masked_array():
    mask = ak.index.Index8(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    content = ak.contents.NumpyArray(np.arange(16))
    layout = ak.contents.ByteMaskedArray(
        mask,
        content,
        False,
    )
    packed = ak.operations.to_packed(layout, highlevel=False)
    assert to_list(packed) == to_list(layout)
    assert len(packed.content) == 8


def test_record():
    a = ak.contents.NumpyArray(np.arange(10))
    b = ak.contents.NumpyArray(np.arange(10) * 2 + 4)
    layout = ak.contents.RecordArray([a, b], None, 5)
    record = layout[4]
    packed = ak.operations.to_packed(record, highlevel=False)
    assert to_list(packed) == to_list(record)
    assert len(packed.array) == 1
    assert to_list(packed.array) == [(4, 12)]
