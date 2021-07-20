# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_numpy_array():
    matrix = np.arange(64).reshape(8, -1)
    layout = ak.layout.NumpyArray(matrix[:, 0])
    assert not layout.iscontiguous

    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert packed.iscontiguous


def test_empty_array():
    layout = ak.layout.EmptyArray()
    assert ak.packed(layout, highlevel=False) is layout


def test_indexed_option_array():
    index = ak.layout.Index64(np.r_[0, -1, 2, -1, 4])
    content = ak.layout.NumpyArray(np.arange(8))
    layout = ak.layout.IndexedOptionArray64(index, content)

    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(layout) == ak.to_list(packed)
    assert isinstance(packed, ak.layout.IndexedOptionArray64)
    assert np.asarray(packed.index).tolist() == [0, -1, 1, -1, 2]
    assert len(packed.content) == 3


def test_indexed_array():
    index = ak.layout.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak.layout.NumpyArray(np.arange(10))
    layout = ak.layout.IndexedArray64(index, content)

    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)

    assert isinstance(packed, ak.layout.NumpyArray)
    assert len(packed) == len(index)


def test_list_array():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.layout.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.layout.Index64(np.array([3, 3, 5, 6, 9]))
    layout = ak.layout.ListArray64(starts, stops, content)

    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert isinstance(packed, ak.layout.ListOffsetArray64)
    assert packed.offsets[0] == 0


def test_list_offset_array():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6]))
    layout = ak.layout.ListOffsetArray64(offsets, content)

    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert isinstance(packed, ak.layout.ListOffsetArray64)
    assert packed.offsets[0] == 0
    assert len(packed.content) == packed.offsets[-1]


def test_unmasked_array():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    layout = ak.layout.UnmaskedArray(content)
    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)


def test_union_array():
    a = ak.layout.NumpyArray(np.arange(4))
    b = ak.layout.NumpyArray(np.arange(4) + 4)
    c = ak.layout.RegularArray(ak.layout.NumpyArray(np.arange(12)), 3)
    layout = ak.layout.UnionArray8_64(
        ak.layout.Index8([1, 1, 2, 2, 0, 0]),
        ak.layout.Index64([0, 1, 0, 1, 0, 1]),
        [a, b, c],
    )
    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    # Check that it merges like contents
    assert len(packed.contents) == 2
    index_0 = np.asarray(packed.index)[np.asarray(packed.tags) == 0]
    assert index_0.tolist() == [0, 1, 2, 3]


def test_record_array():
    a = ak.layout.NumpyArray(np.arange(10))
    b = ak.layout.NumpyArray(np.arange(10) * 2 + 4)
    layout = ak.layout.RecordArray([a, b], None, 5)
    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.contents[0]) == 5
    assert len(packed.contents[1]) == 5


def test_regular_array():
    content = ak.layout.NumpyArray(np.arange(10))
    layout = ak.layout.RegularArray(content, 3)
    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.content) == 9
    assert packed.size == layout.size


def test_bit_masked_aray():
    mask = ak.layout.IndexU8(np.array([0b10101010]))
    content = ak.layout.NumpyArray(np.arange(16))
    layout = ak.layout.BitMaskedArray(mask, content, False, 8, False)
    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.content) == 8


def test_byte_masked_array():
    mask = ak.layout.Index8(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    content = ak.layout.NumpyArray(np.arange(16))
    layout = ak.layout.ByteMaskedArray(
        mask,
        content,
        False,
    )
    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.content) == 8


def test_virtual_array():
    n_called = [0]

    def generate():
        n_called[0] += 1
        return ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))

    generator = ak.layout.ArrayGenerator(
        generate, form=ak.forms.NumpyForm([], 8, "d"), length=5
    )
    layout = ak.layout.VirtualArray(generator)
    assert n_called[0] == 0
    packed = ak.packed(layout, highlevel=False)
    assert n_called[0] == 1

    assert isinstance(packed, ak.layout.NumpyArray)
    assert ak.to_list(packed) == ak.to_list(layout)


def test_partitioned_array():
    index_0 = ak.layout.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content_0 = ak.layout.NumpyArray(np.arange(10))
    content = ak.layout.IndexedArray64(index_0, content_0)
    layout = ak.partitioned([content, content], highlevel=False)
    packed = ak.packed(layout, highlevel=False)

    assert ak.to_list(layout) == ak.to_list(packed)
    assert isinstance(packed, ak.partition.PartitionedArray)

    assert isinstance(packed.partitions[0], ak.layout.NumpyArray)
    assert len(packed.partitions[0]) == len(index_0)

    assert isinstance(packed.partitions[1], ak.layout.NumpyArray)
    assert len(packed.partitions[1]) == len(index_0)


def test_record():
    a = ak.layout.NumpyArray(np.arange(10))
    b = ak.layout.NumpyArray(np.arange(10) * 2 + 4)
    layout = ak.layout.RecordArray([a, b], None, 5)
    record = layout[4]
    packed = ak.packed(record, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(record)
    assert len(packed.array) == 1
    assert ak.to_list(packed.array) == [(4, 12)]
