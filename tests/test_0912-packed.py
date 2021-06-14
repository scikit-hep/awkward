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
    assert packed.index.tolist() == [0, -1, 1, -1, 2]
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


def test_unmasked_array():
    content = ak.layout.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    layout = ak.layout.UnmaskedArray(content)
    packed = ak.packed(layout, highlevel=False)
    assert ak.to_list(packed) == ak.to_list(layout)


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
