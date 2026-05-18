# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pickle

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray


def virtualize(array):
    form, length, container = ak.to_buffers(array)
    new_container = {k: lambda v=v: v for k, v in container.items()}
    return ak.from_buffers(form, length, new_container, highlevel=False)


def test_virtualarray_pickling():
    v = VirtualNDArray(
        Numpy.instance(),
        shape=(5,),
        dtype=np.int64,
        generator=lambda: np.array([1, 2, 3, 4, 5], dtype=np.int64),
    )

    pickled = pickle.dumps(v)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, np.ndarray)
    np.testing.assert_array_equal(unpickled, v.materialize())
    assert unpickled.shape == v.shape
    assert unpickled.dtype == v.dtype
    assert v.is_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_numpy_array_to_packed(recursive):
    matrix = np.arange(64).reshape(8, -1)
    tmp = ak.contents.NumpyArray(matrix[:, 0])

    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    assert layout.is_all_materialized
    assert packed.is_all_materialized
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualNDArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


def test_numpy_array_pickling():
    matrix = np.arange(64).reshape(8, -1)
    tmp = ak.contents.NumpyArray(matrix[:, 0])

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_empty_array_to_packed(recursive):
    tmp = ak.contents.EmptyArray()

    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    assert layout.is_all_materialized
    assert packed.is_all_materialized
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualNDArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


def tesst_empty_array_pickling():
    tmp = ak.contents.EmptyArray()

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_indexed_option_array_to_packed(recursive):
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
        assert isinstance(buffer, VirtualNDArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


def test_indexed_option_array_pickling():
    index = ak.index.Index64(np.r_[0, -1, 2, -1, 4])
    content = ak.contents.NumpyArray(np.arange(8))
    tmp = ak.contents.IndexedOptionArray(index, content)

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_indexed_array_to_packed(recursive):
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
        assert isinstance(buffer, VirtualNDArray)
    for _key, buffer in ak.to_buffers(packed)[2].items():
        assert isinstance(buffer, np.ndarray)


@pytest.mark.parametrize("recursive", [True, False])
def test_list_array_to_packed(recursive):
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
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_list_array_pickling():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    starts = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    stops = ak.index.Index64(np.array([3, 3, 5, 6, 9]))
    tmp = ak.contents.ListArray(starts, stops, content)

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_list_offset_array_to_packed(recursive):
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
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_list_offset_array_pickling():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6]))
    tmp = ak.contents.ListOffsetArray(offsets, content)

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_unmasked_array_to_packed(recursive):
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    tmp = ak.contents.UnmaskedArray(content)

    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_unmasked_array_pickling():
    content = ak.contents.NumpyArray(
        np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    tmp = ak.contents.UnmaskedArray(content)

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_union_array_to_packed(recursive):
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
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_union_array_pickling():
    a = ak.contents.NumpyArray(np.arange(4))
    b = ak.contents.NumpyArray(np.arange(4) + 4)
    c = ak.contents.RegularArray(ak.contents.NumpyArray(np.arange(12)), 3)
    tmp = ak.contents.UnionArray.simplified(
        ak.index.Index8([1, 1, 2, 2, 0, 0]),
        ak.index.Index64([0, 1, 0, 1, 0, 1]),
        [a, b, c],
    )

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_record_array_to_packed(recursive):
    a = ak.contents.NumpyArray(np.arange(10))
    b = ak.contents.NumpyArray(np.arange(10) * 2 + 4)
    tmp = ak.contents.RecordArray([a, b], None, 5)

    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            elif _key == "node2-data":
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_record_array_pickling():
    a = ak.contents.NumpyArray(np.arange(10))
    b = ak.contents.NumpyArray(np.arange(10) * 2 + 4)
    tmp = ak.contents.RecordArray([a, b], None, 5)

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_regular_array_to_packed(recursive):
    content = ak.contents.NumpyArray(np.arange(10))
    tmp = ak.contents.RegularArray(content, 3)

    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_regular_array_pickling():
    content = ak.contents.NumpyArray(np.arange(10))
    tmp = ak.contents.RegularArray(content, 3)

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_bit_masked_array_to_packed(recursive):
    mask = ak.index.IndexU8(np.array([0b10101010]))
    content = ak.contents.NumpyArray(np.arange(16))
    tmp = ak.contents.BitMaskedArray(mask, content, False, 8, False)

    layout = virtualize(tmp)
    assert layout.to_packed(recursive).to_list() == tmp.to_packed(recursive).to_list()
    layout = virtualize(tmp)
    packed = layout.to_packed(recursive)
    for _key, buffer in ak.to_buffers(layout)[2].items():
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            elif _key == "node1-data":
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_bit_masked_array_pickling():
    mask = ak.index.IndexU8(np.array([0b10101010]))
    content = ak.contents.NumpyArray(np.arange(16))
    tmp = ak.contents.BitMaskedArray(mask, content, False, 8, False)

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized


@pytest.mark.parametrize("recursive", [True, False])
def test_byte_masked_array_to_packed(recursive):
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
        assert isinstance(buffer, VirtualNDArray)
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
                assert isinstance(buffer, VirtualNDArray)
            elif _key == "node1-data":
                assert isinstance(buffer, VirtualNDArray)
            else:
                raise ValueError(f"Unexpected key in buffers: {_key}")


def test_byte_masked_array_pickling():
    mask = ak.index.Index8(np.array([1, 0, 1, 0, 1, 0, 1, 0]))
    content = ak.contents.NumpyArray(np.arange(16))
    tmp = ak.contents.ByteMaskedArray(
        mask,
        content,
        False,
    )

    layout = virtualize(tmp)
    assert not layout.is_any_materialized
    pickled = pickle.dumps(layout)
    assert layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.to_list() == tmp.to_list()
    assert unpickled.is_all_materialized

    array = ak.Array(virtualize(tmp))
    assert not array.layout.is_any_materialized
    pickled = pickle.dumps(array)
    assert array.layout.is_all_materialized
    unpickled = pickle.loads(pickled)
    assert unpickled.layout.to_list() == tmp.to_list()
    assert unpickled.layout.is_all_materialized
