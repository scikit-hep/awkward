# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.shape import unknown_length
from awkward._nplikes.virtual import VirtualNDArray


@pytest.mark.parametrize(
    "index_type", [ak.index.Index32, ak.index.IndexU32, ak.index.Index64]
)
@pytest.mark.parametrize(
    "values, indices, expected",
    [([], [], []), ([10], [0], [10]), ([10, 20, 30], [2, 0, 2, 1], [30, 10, 30, 20])],
)
def test_conversion(index_type, values, indices, expected):
    content = ak.contents.NumpyArray(np.array(values, dtype=np.int64))
    index = index_type(indices)
    layout = ak.contents.IndexedArray(index, content, parameters={"label": "kept"})
    result = layout.to_IndexedOptionArray64()

    assert isinstance(result, ak.contents.IndexedOptionArray)
    assert result.index.dtype == np.dtype(np.int64)
    assert result.index.data.tolist() == indices
    assert result.to_list() == expected
    assert result.length == layout.length
    assert result.content is content
    assert result.parameters == layout.parameters == {"label": "kept"}
    assert result.backend is layout.backend
    assert layout.index is index
    assert index.data.tolist() == indices
    assert content.to_list() == values
    if index_type is ak.index.Index64:
        assert result.index is index


@pytest.mark.parametrize(
    "index_type", [ak.index.Index32, ak.index.IndexU32, ak.index.Index64]
)
def test_field_projection(index_type):
    field = ak.contents.IndexedArray(
        index_type([2, 0, 2, 1]), ak.to_layout([10, 20, 30])
    )
    layout = ak.contents.UnmaskedArray(ak.contents.RecordArray([field], ["A"]))
    result = layout["A"]

    assert result.index.dtype == np.dtype(np.int64)
    assert result.to_list() == [30, 10, 30, 20]
    assert result.content is field.content


@pytest.mark.parametrize(
    "index_type", [ak.index.Index32, ak.index.IndexU32, ak.index.Index64]
)
@pytest.mark.parametrize("forget_length", [False, True])
def test_typetracer(index_type, forget_length):
    layout = ak.contents.IndexedArray(
        index_type([2, 0, 2, 1]),
        ak.to_layout([10, 20, 30]),
        parameters={"label": "kept"},
    ).to_typetracer(forget_length=forget_length)
    result = layout.to_IndexedOptionArray64()

    assert result.form.index == "i64"
    assert result.content is layout.content
    assert result.form.content == layout.form.content
    assert result.parameters == layout.parameters
    assert result.backend is layout.backend
    assert ak.backend(result) == "typetracer"
    if forget_length:
        assert result.length is unknown_length
    else:
        assert result.length == 4
    if index_type is ak.index.Index64:
        assert result.index is layout.index


def test_virtual_int64_index():
    def unexpected_materialization():
        pytest.fail("Conversion must not materialize an int64 index or content")

    index = ak.index.Index64(
        VirtualNDArray(
            Numpy.instance(), (4,), np.dtype(np.int64), unexpected_materialization
        )
    )
    content = ak.contents.NumpyArray(
        VirtualNDArray(
            Numpy.instance(), (3,), np.dtype(np.int64), unexpected_materialization
        )
    )
    layout = ak.contents.IndexedArray(index, content)
    result = layout.to_IndexedOptionArray64()

    assert result.index is index
    assert result.content is content
    assert not result.is_any_materialized


@pytest.mark.parametrize("index_type", [ak.index.Index32, ak.index.IndexU32])
def test_conversion_keeps_virtual_content(index_type):
    def unexpected_materialization():
        pytest.fail("Casting the index must not materialize content")

    content = ak.contents.NumpyArray(
        VirtualNDArray(
            Numpy.instance(), (3,), np.dtype(np.int64), unexpected_materialization
        )
    )
    layout = ak.contents.IndexedArray(index_type([2, 0, 2, 1]), content)
    result = layout.to_IndexedOptionArray64()

    assert result.index.dtype == np.dtype(np.int64)
    assert result.index.data.tolist() == [2, 0, 2, 1]
    assert result.content is content
    assert not content.is_any_materialized


def test_option_constructor_rejects_uint32():
    with pytest.raises(TypeError, match=r"dtype in \(int32, int64\)"):
        ak.contents.IndexedOptionArray(ak.index.IndexU32([0]), ak.to_layout([10]))
