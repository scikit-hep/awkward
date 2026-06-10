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
    # The dtype-mapping function (used instead of to_pandas_dtype) must return
    # the correct NumPy dtype for every primitive Arrow type reachable in the
    # DataType fallback branch of popbuffers.
    cases = {
        pyarrow.int8(): np.int8,
        pyarrow.int16(): np.int16,
        pyarrow.int32(): np.int32,
        pyarrow.int64(): np.int64,
        pyarrow.uint8(): np.uint8,
        pyarrow.uint16(): np.uint16,
        pyarrow.uint32(): np.uint32,
        pyarrow.uint64(): np.uint64,
        pyarrow.float16(): np.float16,
        pyarrow.float32(): np.float32,
        pyarrow.float64(): np.float64,
        pyarrow.bool_(): np.bool_,
    }
    for arrow_type, np_type in cases.items():
        assert arrow_to_numpy_dtype(arrow_type) == np.dtype(np_type)


def test_from_arrow_empty_chunked_array():
    # Zero chunks: concatenate([]) would raise; must build a length-zero array.
    array = ak.from_arrow(pyarrow.chunked_array([], type=pyarrow.int64()))
    assert array.to_list() == []
    assert str(array.type) == "0 * ?int64"


def test_from_arrow_all_empty_chunks():
    # Every chunk empty: same length-zero handling.
    array = ak.from_arrow(
        pyarrow.chunked_array(
            [pyarrow.array([], type=pyarrow.float64())], type=pyarrow.float64()
        )
    )
    assert array.to_list() == []


def test_from_arrow_nonempty_chunked_array():
    array = ak.from_arrow(
        pyarrow.chunked_array(
            [pyarrow.array([1, 2]), pyarrow.array([], type=pyarrow.int64())]
        )
    )
    assert array.to_list() == [1, 2]


def test_multifile_parquet_attrs_roundtrip(tmp_path):
    data1 = ak.from_iter([[1, 2, 3], [4, 5]], attrs={"property": "value"})
    data2 = data1 + 1
    ak.to_parquet(data1, os.path.join(tmp_path, "data1.parquet"))
    ak.to_parquet(data2, os.path.join(tmp_path, "data2.parquet"))

    # Single-file path already worked; verify it still does.
    single = ak.from_parquet(os.path.join(tmp_path, "data1.parquet"))
    assert single.attrs == {"property": "value"}

    # Multi-file dataset (spanning >1 file) must preserve attrs too.
    dataset = ak.from_parquet(str(tmp_path))
    assert dataset.attrs == {"property": "value"}
    assert dataset.to_list() == [[1, 2, 3], [4, 5], [2, 3, 4], [5, 6]]
