# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak
from awkward._nplikes.shape import unknown_length

inner_layout = ak.contents.ListOffsetArray(
    ak.index.Index64([0, 3, 4, 4]), ak.contents.NumpyArray(np.arange(4))
)


def test_drop_outer_listoffset():
    layout = ak.contents.ListOffsetArray(ak.index.Index64([0, 2, 2]), inner_layout)
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_list():
    layout = ak.contents.ListArray(
        ak.index.Index64([0, 2]), ak.index.Index64([2, 2]), inner_layout
    )
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_regular():
    layout = ak.contents.RegularArray(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(
                np.arange(12),
            ),
            3,
        ),
        2,
    )
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_unmasked():
    layout = ak.contents.UnmaskedArray(inner_layout)
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_bytemasked():
    layout = ak.contents.ByteMaskedArray(
        ak.index.Index8([1, 0]),
        inner_layout,
        valid_when=True,
    )
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_bitmasked():
    layout = ak.contents.BitMaskedArray(
        ak.index.IndexU8([1, 0]),
        inner_layout,
        valid_when=True,
        length=2,
        lsb_order=True,
    )
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_indexed_option():
    layout = ak.contents.IndexedOptionArray(ak.index.Index64([0, 1]), inner_layout)
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_indexed():
    layout = ak.contents.IndexedArray(ak.index.Index64([0, 1]), inner_layout)
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is not unknown_length
    assert layout_tt.content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    assert layout_tt.content.length is unknown_length
    assert layout_tt.content.content.length is unknown_length


def test_drop_outer_record():
    layout = ak.contents.RecordArray([inner_layout, inner_layout], ["x", "y"])
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    for content in layout_tt.contents:
        assert content.length is not unknown_length
        assert content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    for content in layout_tt.contents:
        assert content.length is not unknown_length
        assert content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    for content in layout_tt.contents:
        assert content.length is unknown_length
        assert content.content.length is unknown_length


def test_drop_outer_union():
    layout = ak.contents.UnionArray(
        ak.index.Index8([0, 0, 1, 1]),
        ak.index.Index64([0, 1, 0, 1]),
        [inner_layout, inner_layout],
    )
    layout_tt = layout.to_typetracer(length_policy="keep")
    assert layout_tt.length is not unknown_length
    for content in layout_tt.contents:
        assert content.length is not unknown_length
        assert content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_outer")
    assert layout_tt.length is unknown_length
    for content in layout_tt.contents:
        assert content.length is not unknown_length
        assert content.content.length is not unknown_length

    layout_tt = layout.to_typetracer(length_policy="drop_recursive")
    assert layout_tt.length is unknown_length
    for content in layout_tt.contents:
        assert content.length is unknown_length
        assert content.content.length is unknown_length
