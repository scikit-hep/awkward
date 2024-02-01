# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak


def test_wrap_index_cupy():
    cp = pytest.importorskip("cupy")
    data = cp.arange(10, dtype=cp.int64)
    index = ak.index.Index64(data)
    other_data = cp.asarray(index)
    assert cp.shares_memory(data, other_data)


def test_wrap_index_numpy():
    data = np.arange(10, dtype=np.int64)
    index = ak.index.Index64(data)
    other_data = np.asarray(index)
    assert np.shares_memory(data, other_data)


def test_wrap_bare_list():
    data = [1, 2, 3, 4, 5]
    index = ak.index.Index64(data)
    other_data = np.asarray(index)
    assert other_data.tolist() == data
