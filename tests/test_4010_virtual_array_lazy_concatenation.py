# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.virtual import VirtualNDArray

nplike = Numpy.instance()


def _virtual(array):
    return VirtualNDArray(
        nplike,
        shape=array.shape,
        dtype=array.dtype,
        generator=lambda: array,
    )


def _virtualize(array):
    form, length, container = ak.to_buffers(array)
    new_container = {k: (lambda v=v: v) for k, v in container.items()}
    return ak.from_buffers(form, length, new_container, highlevel=False)


def _assert_matches(result, expected):
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    materialized = (
        result.materialize() if isinstance(result, VirtualNDArray) else result
    )
    np.testing.assert_array_equal(materialized, expected)


def test_concat_axis0_stays_virtual():
    raw = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5], dtype=np.int64)]
    a = _virtual(raw[0])
    b = _virtual(raw[1])

    result = nplike.concat([a, b])
    expected = np.concatenate(raw, axis=0)

    assert isinstance(result, VirtualNDArray)
    assert not result.is_materialized
    assert not a.is_materialized
    assert not b.is_materialized
    _assert_matches(result, expected)


def test_concat_dtype_promotion():
    raw = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4.5, 5.5], dtype=np.float32),
    ]
    a = _virtual(raw[0])
    b = _virtual(raw[1])

    result = nplike.concat([a, b])
    expected = np.concatenate(raw, axis=0)

    assert isinstance(result, VirtualNDArray)
    assert not result.is_materialized
    _assert_matches(result, expected)


def test_concat_axis_positive():
    raw = [
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.arange(6, 12, dtype=np.int64).reshape(2, 3),
    ]
    a = _virtual(raw[0])
    b = _virtual(raw[1])

    result = nplike.concat([a, b], axis=1)
    expected = np.concatenate(raw, axis=1)

    assert isinstance(result, VirtualNDArray)
    assert not result.is_materialized
    _assert_matches(result, expected)


def test_concat_negative_axis():
    raw = [
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.arange(6, 12, dtype=np.int64).reshape(2, 3),
    ]
    a = _virtual(raw[0])
    b = _virtual(raw[1])

    result = nplike.concat([a, b], axis=-1)
    expected = np.concatenate(raw, axis=-1)

    assert isinstance(result, VirtualNDArray)
    _assert_matches(result, expected)


def test_concat_axis_none_flattens():
    raw = [
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.array([10, 20], dtype=np.int64),
    ]
    a = _virtual(raw[0])
    b = _virtual(raw[1])

    result = nplike.concat([a, b], axis=None)
    expected = np.concatenate(raw, axis=None)

    assert isinstance(result, VirtualNDArray)
    assert not result.is_materialized
    _assert_matches(result, expected)


def test_concat_mixed_virtual_and_concrete():
    raw = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5], dtype=np.int64)]
    a = _virtual(raw[0])
    b = raw[1]

    result = nplike.concat([a, b])
    expected = np.concatenate(raw, axis=0)

    assert isinstance(result, VirtualNDArray)
    assert not result.is_materialized
    assert not a.is_materialized
    _assert_matches(result, expected)


def test_concat_all_concrete_falls_through():
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([4, 5], dtype=np.int64)

    result = nplike.concat([a, b])

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])


def test_concat_materialized_virtual_falls_through():
    a = _virtual(np.array([1, 2, 3], dtype=np.int64))
    b = _virtual(np.array([4, 5], dtype=np.int64))
    a.materialize()
    b.materialize()

    result = nplike.concat([a, b])

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])


def test_concat_shape_mismatch_message():
    a = _virtual(np.empty((5, 2), dtype=np.float64))
    b = _virtual(np.empty((2, 3), dtype=np.float64))

    with pytest.raises(
        ValueError,
        match=r"along dimension 1, the array at index 0 has size 2 and the array at index 1 has size 3",
    ):
        nplike.concat([a, b], axis=0)


def test_concat_shape_mismatch_reports_first():
    a = _virtual(np.empty((5, 2), dtype=np.float64))
    b = _virtual(np.empty((2, 3), dtype=np.float64))
    c = _virtual(np.empty((3, 7), dtype=np.float64))

    with pytest.raises(
        ValueError,
        match=r"the array at index 0 has size 2 and the array at index 1 has size 3",
    ):
        nplike.concat([a, b, c], axis=0)


def test_concat_does_not_materialize_inputs():
    a = _virtual(np.array([1, 2, 3], dtype=np.int64))
    b = _virtual(np.array([4, 5], dtype=np.int64))

    result = nplike.concat([a, b])

    assert not result.is_materialized
    assert not a.is_materialized
    assert not b.is_materialized

    result.materialize()

    assert result.is_materialized
    assert a.is_materialized
    assert b.is_materialized


def test_concat_axis0_numpyarray_stays_virtual():
    raw = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5])
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    assert not result.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_listoffsetarray_keeps_content_virtual():
    raw = ak.Array([[1, 2, 3], [], [4, 5]])
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    assert result.offsets.is_all_materialized
    assert not result.content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_recordarray_stays_virtual():
    raw = ak.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    assert not result.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_record_of_jagged_fields():
    raw = ak.Array(
        [
            {"x": [1, 2, 3], "y": [1.1]},
            {"x": [], "y": [2.2, 3.3]},
            {"x": [4, 5], "y": []},
        ]
    )
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    for field_name in ("x", "y"):
        field = result[field_name]
        assert field.offsets.is_all_materialized
        assert not field.content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_record_mixed_jagged_and_flat():
    raw = ak.Array(
        [
            {"x": [1, 2, 3], "y": 1.1},
            {"x": [], "y": 2.2},
            {"x": [4, 5], "y": 3.3},
        ]
    )
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    assert result["x"].offsets.is_all_materialized
    assert not result["x"].content.is_any_materialized
    assert not result["y"].is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_option_with_none():
    raw = ak.Array([1, None, 2, None, 3])
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    assert result.index.is_all_materialized
    assert not result.content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_jagged_with_none():
    raw = ak.Array([[1, 2], None, [3, 4, 5], [], None])
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    assert result.index.is_all_materialized
    assert result.content.offsets.is_all_materialized
    assert not result.content.content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_union():
    raw = ak.concatenate([ak.Array([1, 2, 3]), ak.Array(["a", "b"])])
    a = _virtualize(raw)
    b = _virtualize(raw)

    assert not a.is_any_materialized
    assert not b.is_any_materialized

    result = ak.concatenate([a, b], axis=0, highlevel=False)

    assert result.tags.is_all_materialized
    assert result.index.is_all_materialized
    assert not result.contents[0].is_any_materialized
    assert result.contents[1].offsets.is_all_materialized
    assert not result.contents[1].content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw, raw], axis=0))


def test_concat_axis0_many_numpyarrays():
    raw = ak.Array([1.1, 2.2, 3.3])
    inputs = [_virtualize(raw) for _ in range(5)]

    for x in inputs:
        assert not x.is_any_materialized

    result = ak.concatenate(inputs, axis=0, highlevel=False)

    assert not result.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw] * 5, axis=0))


def test_concat_axis0_many_listoffsetarrays():
    raw = ak.Array([[1, 2, 3], [], [4, 5]])
    inputs = [_virtualize(raw) for _ in range(4)]

    for x in inputs:
        assert not x.is_any_materialized

    result = ak.concatenate(inputs, axis=0, highlevel=False)

    assert result.offsets.is_all_materialized
    assert not result.content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw] * 4, axis=0))


def test_concat_axis0_deeply_nested_lists():
    raw = ak.Array([[[1, 2], [3]], [], [[4], [5, 6, 7]]])
    inputs = [_virtualize(raw) for _ in range(3)]

    for x in inputs:
        assert not x.is_any_materialized

    result = ak.concatenate(inputs, axis=0, highlevel=False)

    assert result.offsets.is_all_materialized
    assert result.content.offsets.is_all_materialized
    assert not result.content.content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw] * 3, axis=0))


def test_concat_axis0_with_empty_input():
    raw_full = ak.Array([[1, 2], [3, 4, 5]])
    raw_empty = raw_full[2:2]
    a = _virtualize(raw_full)
    b = _virtualize(raw_empty)
    c = _virtualize(raw_full)

    for x in (a, b, c):
        assert not x.is_any_materialized

    result = ak.concatenate([a, b, c], axis=0, highlevel=False)

    assert result.offsets.is_all_materialized
    assert not result.content.is_any_materialized
    assert ak.array_equal(
        result, ak.concatenate([raw_full, raw_empty, raw_full], axis=0)
    )


def test_concat_axis0_regulararray():
    raw = ak.to_regular(ak.Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), axis=1)
    inputs = [_virtualize(raw) for _ in range(3)]

    for x in inputs:
        assert not x.is_any_materialized

    result = ak.concatenate(inputs, axis=0, highlevel=False)

    assert not result.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw] * 3, axis=0))


def test_concat_axis0_many_records_of_jagged():
    raw = ak.Array(
        [
            {"x": [1, 2], "y": [10.0, 20.0]},
            {"x": [3, 4, 5], "y": []},
            {"x": [], "y": [30.0]},
        ]
    )
    inputs = [_virtualize(raw) for _ in range(4)]

    for x in inputs:
        assert not x.is_any_materialized

    result = ak.concatenate(inputs, axis=0, highlevel=False)

    for field_name in ("x", "y"):
        field = result[field_name]
        assert field.offsets.is_all_materialized
        assert not field.content.is_any_materialized
    assert ak.array_equal(result, ak.concatenate([raw] * 4, axis=0))
