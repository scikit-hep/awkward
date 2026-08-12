# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import os

import numpy as np
import pytest

import awkward as ak

pyarrow = pytest.importorskip("pyarrow")

from awkward._connect.pyarrow.conversions import arrow_to_numpy_dtype  # noqa: E402


def test_from_arrow_primitive():
    # Primitive Arrow arrays must convert without importing pandas (pyarrow >= 22
    # hard-imports pandas in DataType.to_pandas_dtype()).
    array = ak.from_arrow(pyarrow.array([1.1, 2.2, 3.3]))
    assert array.to_list() == [1.1, 2.2, 3.3]

    array = ak.from_arrow(pyarrow.array([1, 2, 3], type=pyarrow.int32()))
    assert array.to_list() == [1, 2, 3]


def test_arrow_to_numpy_dtype_mapping():
    # The dtype-mapping function must return the correct NumPy dtype for
    # representative primitive Arrow types (signed, unsigned, float, bool).
    assert arrow_to_numpy_dtype(pyarrow.int32()) == np.dtype(np.int32)
    assert arrow_to_numpy_dtype(pyarrow.uint64()) == np.dtype(np.uint64)
    assert arrow_to_numpy_dtype(pyarrow.float32()) == np.dtype(np.float32)
    assert arrow_to_numpy_dtype(pyarrow.bool_()) == np.dtype(np.bool_)


def test_from_arrow_empty_chunked_array():
    # Zero chunks: concatenate([]) would raise; must build a length-zero array.
    array = ak.from_arrow(pyarrow.chunked_array([], type=pyarrow.int64()))
    assert array.to_list() == []
    assert str(array.type) == "0 * ?int64"


def test_multifile_parquet_attrs_roundtrip(tmp_path):
    data1 = ak.from_iter([[1, 2, 3], [4, 5]], attrs={"property": "value"})
    data2 = data1 + 1
    ak.to_parquet(data1, os.path.join(tmp_path, "data1.parquet"))
    ak.to_parquet(data2, os.path.join(tmp_path, "data2.parquet"))

    # Multi-file dataset must preserve attrs from per-file metadata.
    dataset = ak.from_parquet(str(tmp_path))
    assert dataset.attrs == {"property": "value"}
    assert dataset.to_list() == [[1, 2, 3], [4, 5], [2, 3, 4], [5, 6]]
