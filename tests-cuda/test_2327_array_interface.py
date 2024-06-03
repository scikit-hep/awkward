# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest

import awkward as ak


def test_wrap_index_cupy():
    cp = pytest.importorskip("cupy")
    data = cp.arange(10, dtype=cp.int64)
    index = ak.index.Index64(data)
    other_data = cp.asarray(index)
    result = cp.shares_memory(data, other_data)
    assert result is True
