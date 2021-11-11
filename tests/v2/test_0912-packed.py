# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_numpy_array():
    matrix = np.arange(64).reshape(8, -1)
    layout = ak._v2.contents.NumpyArray(matrix[:, 0])
    assert not layout.is_contiguous

    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert packed.is_contiguous


def test_empty_array():
    layout = ak._v2.contents.EmptyArray()
    assert ak._v2.operations.structure.packed(layout, highlevel=False) is layout


def test_indexed_option_array():
    index = ak._v2.index.Index64(np.r_[0, -1, 2, -1, 4])
    content = ak._v2.contents.NumpyArray(np.arange(8))
    layout = ak._v2.contents.IndexedOptionArray(index, content)

    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(layout) == ak.to_list(packed)
    assert isinstance(packed, ak._v2.contents.IndexedOptionArray)
    assert np.asarray(packed.index).tolist() == [0, -1, 1, -1, 2]
    assert len(packed.content) == 3


def test_indexed_array():
    index = ak._v2.index.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak._v2.contents.NumpyArray(np.arange(10))
    layout = ak._v2.contents.IndexedArray(index, content)

    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)

    assert isinstance(packed, ak._v2.contents.NumpyArray)
    assert len(packed) == len(index)


def test_list_array():
    content = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak._v2.index.Index64(np.array([3, 3, 5, 6, 9]))
    layout = ak._v2.contents.ListArray(starts, stops, content)

    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert isinstance(packed, ak._v2.contents.ListOffsetArray)
    assert packed.offsets[0] == 0


def test_list_offset_array():
    content = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak._v2.index.Index64(np.array([0, 3, 3, 5, 6]))
    layout = ak._v2.contents.ListOffsetArray(offsets, content)

    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert isinstance(packed, ak._v2.contents.ListOffsetArray)
    assert packed.offsets[0] == 0
    assert len(packed.content) == packed.offsets[-1]


def test_unmasked_array():
    content = ak._v2.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    layout = ak._v2.contents.UnmaskedArray(content)
    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)


def test_union_array():
    a = ak._v2.contents.NumpyArray(np.arange(4))
    b = ak._v2.contents.NumpyArray(np.arange(4) + 4)
    c = ak._v2.contents.RegularArray(ak._v2.contents.NumpyArray(np.arange(12)), 3)
    layout = ak._v2.contents.UnionArray(
        ak._v2.index.Index8([1, 1, 2, 2, 0, 0]),
        ak._v2.index.Index64([0, 1, 0, 1, 0, 1]),
        [a, b, c],
    )
    packed = ak._v2.operations.structure.packed(
        layout.simplify_uniontype(), highlevel=False
    )
    assert ak.to_list(packed) == ak.to_list(layout)
    # Check that it merges like contents
    assert len(packed.contents) == 2
    index_0 = np.asarray(packed.index)[np.asarray(packed.tags) == 0]
    assert index_0.tolist() == [0, 1, 2, 3]


def test_record_array():
    a = ak._v2.contents.NumpyArray(np.arange(10))
    b = ak._v2.contents.NumpyArray(np.arange(10) * 2 + 4)
    layout = ak._v2.contents.RecordArray([a, b], None, 5)
    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.contents[0]) == 5
    assert len(packed.contents[1]) == 5


def test_regular_array():
    content = ak._v2.contents.NumpyArray(np.arange(10))
    layout = ak._v2.contents.RegularArray(content, 3)
    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.content) == 9
    assert packed.size == layout.size


def test_bit_masked_aray():
    mask = ak._v2.index.IndexU8(np.array([0b10101010]))
    content = ak._v2.contents.NumpyArray(np.arange(16))
    layout = ak._v2.contents.BitMaskedArray(mask, content, False, 8, False)
    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.content) == 8


def test_byte_masked_array():
    mask = ak._v2.index.Index8(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    content = ak._v2.contents.NumpyArray(np.arange(16))
    layout = ak._v2.contents.ByteMaskedArray(
        mask,
        content,
        False,
    )
    packed = ak._v2.operations.structure.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)
    assert len(packed.content) == 8


def test_record():
    a = ak._v2.contents.NumpyArray(np.arange(10))
    b = ak._v2.contents.NumpyArray(np.arange(10) * 2 + 4)
    layout = ak._v2.contents.RecordArray([a, b], None, 5)
    record = layout[4]
    packed = ak._v2.operations.structure.packed(record, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(record)
    assert len(packed.array) == 1
    assert ak.to_list(packed.array) == [(4, 12)]
