# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_numpy_array(forget_length, recursive):
    matrix = np.arange(64).reshape(8, -1)
    layout = ak.contents.NumpyArray(matrix[:, 0])
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_empty_array(forget_length, recursive):
    layout = ak.contents.EmptyArray()
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_indexed_option_array(forget_length, recursive):
    index = ak.index.Index64(np.r_[0, -1, 2, -1, 4])
    content = ak.contents.NumpyArray(np.arange(8))
    layout = ak.contents.IndexedOptionArray(index, content)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_indexed_array(forget_length, recursive):
    index = ak.index.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak.contents.NumpyArray(np.arange(10))
    layout = ak.contents.IndexedArray(index, content)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_list_array(forget_length, recursive):
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
    layout = ak.contents.ListArray(starts, stops, content)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_list_offset_array(forget_length, recursive):
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    layout = ak.contents.ListOffsetArray(offsets, content)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_unmasked_array(forget_length, recursive):
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    layout = ak.contents.UnmaskedArray(content)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_union_array(forget_length, recursive):
    a = ak.contents.NumpyArray(np.arange(4))
    b = ak.contents.NumpyArray(np.arange(4) + 4)
    c = ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(12)), 3)
    layout = ak.contents.UnionArray.simplified(
        ak.index.Index8([1, 1, 2, 2, 0, 0]),
        ak.index.Index64([0, 1, 0, 1, 0, 1]),
        [a, b, c],
    )
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_record_array(forget_length, recursive):
    a = ak.contents.NumpyArray(np.arange(10))
    b = ak.contents.NumpyArray(np.arange(10) * 2 + 4)
    layout = ak.contents.RecordArray([a, b], None, 5)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_regular_array(forget_length, recursive):
    content = ak.contents.NumpyArray(np.arange(10))
    layout = ak.contents.RegularArray(content, 3)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_bit_masked_array(forget_length, recursive):
    mask = ak.index.IndexU8(np.array([0b10101010]))
    content = ak.contents.NumpyArray(np.arange(16))
    layout = ak.contents.BitMaskedArray(mask, content, False, 8, False)
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )


@pytest.mark.parametrize("forget_length", [True, False])
@pytest.mark.parametrize("recursive", [True, False])
def test_byte_masked_array(forget_length, recursive):
    mask = ak.index.Index8(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    content = ak.contents.NumpyArray(np.arange(16))
    layout = ak.contents.ByteMaskedArray(
        mask,
        content,
        False,
    )
    assert (
        layout.to_typetracer(forget_length).to_packed(recursive).form
        == layout.to_packed(recursive).form
    )
