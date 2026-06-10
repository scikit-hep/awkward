# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward.index import Index8, Index64


def test_listoffsetarray_rejects_wrong_index_dtype():
    content = ak.contents.NumpyArray(np.arange(10, dtype=np.float64))
    with pytest.raises(TypeError, match="must be an Index with dtype"):
        ak.contents.ListOffsetArray(Index8(np.array([0, 3, 5], dtype=np.int8)), content)
    # valid construction still works
    ak.contents.ListOffsetArray(
        Index64(np.array([0, 3, 5, 10], dtype=np.int64)), content
    )


def test_listarray_rejects_wrong_starts_dtype():
    content = ak.contents.NumpyArray(np.arange(10, dtype=np.float64))
    with pytest.raises(TypeError, match="'starts' must be an Index with dtype"):
        ak.contents.ListArray(
            Index8(np.array([0, 3], dtype=np.int8)),
            Index8(np.array([3, 5], dtype=np.int8)),
            content,
        )
    # valid construction still works
    ak.contents.ListArray(
        Index64(np.array([0, 3], dtype=np.int64)),
        Index64(np.array([3, 5], dtype=np.int64)),
        content,
    )


def test_unionarray_rejects_wrong_index_dtype():
    content = ak.contents.NumpyArray(np.arange(10, dtype=np.float64))
    with pytest.raises(TypeError, match="'index' must be an Index with dtype"):
        ak.contents.UnionArray(
            Index8(np.array([0, 0], dtype=np.int8)),
            Index8(np.array([0, 1], dtype=np.int8)),
            [content, content],
        )
    # valid construction still works
    ak.contents.UnionArray(
        Index8(np.array([0, 0], dtype=np.int8)),
        Index64(np.array([0, 1], dtype=np.int64)),
        [content, content],
    )


def test_indexedarray_project_mask_length_mismatch():
    content = ak.contents.NumpyArray(np.arange(5, dtype=np.float64))
    layout = ak.contents.IndexedArray(
        Index64(np.array([0, 1, 2, 3, 4], dtype=np.int64)), content
    )
    mask = ak.index.Index8(np.array([0, 1, 1], dtype=np.int8))
    with pytest.raises(ValueError, match="mask length"):
        layout.project(mask=mask)


def test_indexedoptionarray_project_mask_length_mismatch():
    content = ak.contents.NumpyArray(np.arange(5, dtype=np.float64))
    layout = ak.contents.IndexedOptionArray(
        Index64(np.array([0, 1, 2, 3, 4], dtype=np.int64)), content
    )
    mask = ak.index.Index8(np.array([0, 1, 1], dtype=np.int8))
    with pytest.raises(ValueError, match="mask length"):
        layout.project(mask=mask)


def test_record_at_must_be_integer_message():
    array = ak.contents.RecordArray(
        [ak.contents.NumpyArray(np.arange(3, dtype=np.float64))], ["x"]
    )
    with pytest.raises(TypeError, match="'at' must be an integer, not 'notint'"):
        ak.record.Record(array, "notint")


def test_numpyarray_validity_error_mentions_strides():
    base = np.arange(40, dtype=np.int64)
    arr = np.lib.stride_tricks.as_strided(
        base.view(np.int32), shape=(3, 2), strides=(5, 4)
    )
    layout = ak.contents.NumpyArray(arr)
    message = layout._validity_error("layout")
    assert "strides[" in message
    assert "shape[" not in message


def test_unmaskedarray_is_equal_to_non_option():
    content = ak.contents.NumpyArray(np.arange(5, dtype=np.float64))
    unmasked = ak.contents.UnmaskedArray(content)
    # comparing against a non-option layout must not raise; just returns False
    assert unmasked.is_equal_to(content) is False
    assert unmasked.is_equal_to(ak.contents.UnmaskedArray(content)) is True


def test_unmaskedarray_is_equal_to_respects_parameters():
    content = ak.contents.NumpyArray(np.arange(5, dtype=np.float64))
    a = ak.contents.UnmaskedArray(content)
    b = ak.contents.UnmaskedArray(content).copy(parameters={"__array__": "foo"})
    assert a.is_equal_to(b, all_parameters=True) is False


def test_listoffsetarray_getitem_next_slice_branch():
    content = ak.contents.NumpyArray(np.arange(10, dtype=np.float64))
    layout = ak.contents.ListOffsetArray(
        Index64(np.array([0, 3, 5, 10], dtype=np.int64)), content
    )
    out = layout._getitem_next(slice(1, None), (), None)
    assert ak.Array(out).to_list() == [[1.0, 2.0], [4.0], [6.0, 7.0, 8.0, 9.0]]


def test_listoffsetarray_getitem_next_array_branch():
    content = ak.contents.NumpyArray(np.arange(10, dtype=np.float64))
    layout = ak.contents.ListOffsetArray(
        Index64(np.array([0, 3, 5, 10], dtype=np.int64)), content
    )
    out = layout._getitem_next(Index64(np.array([0], dtype=np.int64)), (), None)
    assert ak.Array(out).to_list() == [[0.0], [3.0], [5.0]]


def test_listoffsetarray_unique_axis():
    array = ak.Array([[3, 1, 1, 2], [5, 5], []])
    # exercises the maxnextparents scalar path in ListOffsetArray._unique
    assert ak.sort(array).to_list() == [[1, 1, 2, 3], [5, 5], []]


def test_regulararray_to_arrow_bytestring_roundtrip():
    pa = pytest.importorskip("pyarrow")
    # A genuine RegularArray (not ListOffsetArray) of fixed-size bytestrings,
    # exercising RegularArray._to_arrow's py_buffer path.
    content = ak.contents.NumpyArray(
        np.frombuffer(b"abcdefghi", dtype=np.uint8), parameters={"__array__": "byte"}
    )
    layout = ak.contents.RegularArray(
        content, 3, parameters={"__array__": "bytestring"}
    )
    result = ak.to_arrow(ak.Array(layout))
    assert isinstance(result, pa.Array)
    assert ak.from_arrow(result).to_list() == [b"abc", b"def", b"ghi"]


def test_regulararray_to_backend_array():
    # exercises the known_data assertion in RegularArray._to_backend_array
    content = ak.contents.NumpyArray(np.arange(6, dtype=np.float64))
    layout = ak.contents.RegularArray(content, 2)
    out = layout.to_backend_array()
    assert out.tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]


def test_recordarray_to_backend_array_empty_record():
    layout = ak.contents.RecordArray([], None, length=3)
    out = layout.to_backend_array()
    assert len(out) == 3
    assert out.dtype.names in (None, ())
