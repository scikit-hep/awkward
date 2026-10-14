# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray


@pytest.mark.parametrize(
    "index_type", [ak.index.Index32, ak.index.IndexU32, ak.index.Index64]
)
def test_indexedarray_conversion(index_type):
    layout = ak.contents.IndexedArray(
        index_type([2, 0, 2, 1]), ak.to_layout([10, 20, 30])
    )
    result = layout.to_IndexedOptionArray64()
    assert result.index.dtype == np.dtype(np.int64)
    assert result.to_list() == [30, 10, 30, 20]


def issue_layout():
    return ak.contents.UnmaskedArray(
        ak.contents.RecordArray(
            [
                ak.contents.IndexedArray(
                    ak.index.IndexU32(np.zeros(3, dtype=np.uint32)),
                    ak.contents.NumpyArray(np.array([False, False])),
                )
            ],
            ["A"],
        )
    )


def test_field_projection():
    assert issue_layout()["A"].to_list() == [False, False, False]


def test_arrow_roundtrip():
    pytest.importorskip("pyarrow")
    table = ak.to_arrow_table(ak.Array(issue_layout()))
    expected = [{"A": False}] * 3
    assert table.to_pylist() == expected
    assert ak.from_arrow(table).to_list() == expected


def test_virtual_int64_index():
    def unexpected_materialization():
        pytest.fail("Converting an int64 index must not materialize its buffer")

    index = ak.index.Index64(
        VirtualNDArray(
            Numpy.instance(), (3,), np.dtype(np.int64), unexpected_materialization
        )
    )
    layout = ak.contents.IndexedArray(index, ak.to_layout([10, 20, 30]))
    assert layout.to_IndexedOptionArray64().index is index
