# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray


def test_virtualarray_byteswap_is_lazy_and_remains_virtual_until_materialized(
    monkeypatch,
):
    nplike = Numpy.instance()
    counts = {"generator": 0, "byteswap": 0}

    original_byteswap = nplike.byteswap

    def tracked_byteswap(array):
        counts["byteswap"] += 1
        return original_byteswap(array)

    monkeypatch.setattr(nplike, "byteswap", tracked_byteswap)

    def generator():
        counts["generator"] += 1
        return np.array([1, 2, 3], dtype=np.int32)

    virtual = VirtualNDArray(
        nplike,
        shape=(3,),
        dtype=np.dtype(np.int32),
        generator=generator,
        buffer_key="node0-data",
    )

    swapped = nplike.byteswap(virtual)

    assert isinstance(swapped, VirtualNDArray)
    assert not virtual.is_materialized
    assert not swapped.is_materialized
    assert swapped.buffer_key == virtual.buffer_key
    assert swapped.dtype == virtual.dtype
    assert swapped.shape == virtual.shape
    assert counts["generator"] == 0
    assert counts["byteswap"] == 1

    expected = np.array([1, 2, 3], dtype=np.int32).byteswap()
    np.testing.assert_array_equal(swapped.materialize(), expected)

    assert swapped.is_materialized
    assert counts["generator"] == 1
    assert counts["byteswap"] == 2


def test_virtualarray_byteswap_from_materialized_source_stays_virtual(monkeypatch):
    nplike = Numpy.instance()
    counts = {"generator": 0, "byteswap": 0}

    original_byteswap = nplike.byteswap

    def tracked_byteswap(array):
        counts["byteswap"] += 1
        return original_byteswap(array)

    monkeypatch.setattr(nplike, "byteswap", tracked_byteswap)

    def generator():
        counts["generator"] += 1
        return np.array([10, 20, 30], dtype=np.int64)

    virtual = VirtualNDArray(
        nplike,
        shape=(3,),
        dtype=np.dtype(np.int64),
        generator=generator,
        buffer_key="node1-data",
    )

    np.testing.assert_array_equal(
        virtual.materialize(), np.array([10, 20, 30], dtype=np.int64)
    )
    assert virtual.is_materialized
    assert counts["generator"] == 1
    assert counts["byteswap"] == 0

    swapped = nplike.byteswap(virtual)

    assert isinstance(swapped, VirtualNDArray)
    assert not swapped.is_materialized
    assert swapped.buffer_key == virtual.buffer_key
    assert swapped.dtype == virtual.dtype
    assert swapped.shape == virtual.shape
    assert counts["generator"] == 1
    assert counts["byteswap"] == 1

    expected = np.array([10, 20, 30], dtype=np.int64).byteswap()
    np.testing.assert_array_equal(swapped.materialize(), expected)

    assert swapped.is_materialized
    assert counts["generator"] == 1
    assert counts["byteswap"] == 2
