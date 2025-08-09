# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.virtual import VirtualArray


def virtualize(array):
    form, length, container = ak.to_buffers(array)
    new_container = {k: lambda v=v: v for k, v in container.items()}
    return ak.from_buffers(form, length, new_container, highlevel=False)


@pytest.mark.parametrize("recursive", [True, False])
def test_numpy_array(recursive):
    matrix = np.arange(64).reshape(8, -1)
    tmp = ak.contents.NumpyArray(matrix[:, 0])
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    assert layout.is_all_materialized
    assert packed.is_all_materialized
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


@pytest.mark.parametrize("recursive", [True, False])
def test_empty_array(recursive):
    tmp = ak.contents.EmptyArray()
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    assert layout.is_all_materialized
    assert packed.is_all_materialized
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


@pytest.mark.parametrize("recursive", [True, False])
def test_indexed_option_array(recursive):
    index = ak.index.Index64(np.r_[0, -1, 2, -1, 4])
    content = ak.contents.NumpyArray(np.arange(8))
    tmp = ak.contents.IndexedOptionArray(index, content)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    assert layout.is_all_materialized
    assert packed.is_all_materialized
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


@pytest.mark.parametrize("recursive", [True, False])
def test_indexed_array(recursive):
    index = ak.index.Index64(np.array([0, 1, 2, 3, 6, 7, 8]))
    content = ak.contents.NumpyArray(np.arange(10))
    tmp = ak.contents.IndexedArray(index, content)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    assert layout.is_all_materialized
    assert packed.is_all_materialized
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


@pytest.mark.parametrize("recursive", [True, False])
def test_list_array(recursive):
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
    tmp = ak.contents.ListArray(starts, stops, content)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert layout.is_any_materialized
        assert packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node0-offsets":
                assert isinstance(buffer, np.ndarray)
            elif _key == "node1-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


@pytest.mark.parametrize("recursive", [True, False])
def test_list_offset_array(recursive):
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    tmp = ak.contents.ListOffsetArray(offsets, content)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert layout.is_any_materialized
        assert packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node0-offsets":
                assert isinstance(buffer, np.ndarray)
            elif _key == "node1-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


@pytest.mark.parametrize("recursive", [True, False])
def test_unmasked_array(recursive):
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    tmp = ak.contents.UnmaskedArray(content)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert not layout.is_any_materialized
        assert not packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node1-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


@pytest.mark.parametrize("recursive", [True, False])
def test_union_array(recursive):
    a = ak.contents.NumpyArray(np.arange(4))
    b = ak.contents.NumpyArray(np.arange(4) + 4)
    c = ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(12)), 3)
    tmp = ak.contents.UnionArray.simplified(
        ak.index.Index8([1, 1, 2, 2, 0, 0]),
        ak.index.Index64([0, 1, 0, 1, 0, 1]),
        [a, b, c],
    )
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert layout.is_any_materialized
        assert packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node0-tags":
                assert isinstance(buffer, np.ndarray)
            elif _key == "node0-index":
                assert isinstance(buffer, np.ndarray)
            elif _key == "node1-data":
                assert isinstance(buffer, np.ndarray)
            elif _key == "node3-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


@pytest.mark.parametrize("recursive", [True, False])
def test_record_array(recursive):
    a = ak.contents.NumpyArray(np.arange(10))
    b = ak.contents.NumpyArray(np.arange(10) * 2 + 4)
    tmp = ak.contents.RecordArray([a, b], None, 5)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert not layout.is_any_materialized
        assert not packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node1-data":
                assert isinstance(buffer, VirtualArray)
            elif _key == "node2-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


@pytest.mark.parametrize("recursive", [True, False])
def test_regular_array(recursive):
    content = ak.contents.NumpyArray(np.arange(10))
    tmp = ak.contents.RegularArray(content, 3)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert not layout.is_any_materialized
        assert not packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node1-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


@pytest.mark.parametrize("recursive", [True, False])
def test_bit_masked_array(recursive):
    mask = ak.index.IndexU8(np.array([0b10101010]))
    content = ak.contents.NumpyArray(np.arange(16))
    tmp = ak.contents.BitMaskedArray(mask, content, False, 8, False)
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert not layout.is_any_materialized
        assert not packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node0-mask":
                assert isinstance(buffer, VirtualArray)
            elif _key == "node1-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


@pytest.mark.parametrize("recursive", [True, False])
def test_byte_masked_array(recursive):
    mask = ak.index.Index8(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    content = ak.contents.NumpyArray(np.arange(16))
    tmp = ak.contents.ByteMaskedArray(
        mask,
        content,
        False,
    )
    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualArray)
    if recursive:
        assert layout.is_all_materialized
        assert packed.is_all_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            assert isinstance(buffer, np.ndarray)
    else:
        assert not layout.is_all_materialized
        assert not packed.is_all_materialized
        assert not layout.is_any_materialized
        assert not packed.is_any_materialized
        for _key, buffer in ak.to_buffers(packed)[2].items():
            if _key == "node0-mask":
                assert isinstance(buffer, VirtualArray)
            elif _key == "node1-data":
                assert isinstance(buffer, VirtualArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")
