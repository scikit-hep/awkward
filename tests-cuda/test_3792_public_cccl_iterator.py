# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Public entry point ``ak.cuda.to_cccl_iterator`` (needs a GPU).

Promotes the internal ``awkward_to_cccl_iterator`` buffer-to-iterator builder
to a supported API so external cuda.compute code stops hand-extracting buffers
from ``ak.to_buffers``.  See the follow-up plan, Phase 3.
"""

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


def test_symbol_is_public():
    assert "to_cccl_iterator" in ak.cuda.__all__
    assert callable(ak.cuda.to_cccl_iterator)


def test_list_array_metadata():
    arr = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]], backend="cuda")
    _it, meta = ak.cuda.to_cccl_iterator(arr)

    assert set(meta) == {"form", "buffers", "offsets", "length", "count"}
    assert meta["length"] == 3
    assert meta["count"] == 9
    assert meta["offsets"] is not None
    assert cp.asarray(meta["offsets"]).tolist() == [0, 3, 5, 9]


def test_flat_numpy_array_has_no_offsets():
    arr = ak.Array(np.arange(5.0), backend="cuda")
    it, meta = ak.cuda.to_cccl_iterator(arr)
    assert meta["offsets"] is None
    # a bare NumpyArray lowers to a CuPy buffer
    assert cp.asarray(it).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_cpu_array_is_moved_to_device():
    # A non-CUDA array is accepted and moved to the device (documented).
    arr = ak.Array([[1, 2], [3]])  # cpu backend
    _it, meta = ak.cuda.to_cccl_iterator(arr)
    assert meta["count"] == 3
    assert cp.asarray(meta["offsets"]).tolist() == [0, 2, 3]


def test_dtype_cast():
    arr = ak.Array([[1, 2, 3], [4, 5]], backend="cuda")
    it, _meta = ak.cuda.to_cccl_iterator(arr, dtype=np.float32)
    # the leaf buffer is cast while building the iterator
    assert cp.asarray(it).dtype == cp.float32


def test_record_array_builds_zip_iterator():
    from cuda.compute import ZipIterator

    arr = ak.Array([{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}], backend="cuda")
    it, _meta = ak.cuda.to_cccl_iterator(arr)
    assert isinstance(it, ZipIterator)


def test_matches_internal_helper():
    from awkward._connect.cuda.helpers import awkward_to_cccl_iterator

    arr = ak.Array([[1.0, 2.0], [3.0, 4.0, 5.0]], backend="cuda")
    _, meta_public = ak.cuda.to_cccl_iterator(arr)
    _, meta_internal = awkward_to_cccl_iterator(arr)
    assert meta_public["count"] == meta_internal["count"]
    assert (
        cp.asarray(meta_public["offsets"]).tolist()
        == cp.asarray(meta_internal["offsets"]).tolist()
    )
