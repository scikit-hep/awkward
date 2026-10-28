# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
from awkward._backends.cupy import CupyBackend

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_flatten(dtype, empty):
    with cp.cuda.Stream(non_blocking=True) as stream:
        offsets = [
            cp.asarray([2, 4, 4, 7], dtype=cp.int64),
            cp.asarray([1, 3, 6], dtype=cp.int64),
        ]
        pointers = cp.asarray([x.data.ptr for x in offsets], dtype=cp.intp)
        tags = cp.asarray([] if empty else [1, 0, 1, 0], dtype=cp.int8)
        index = cp.asarray([] if empty else [1, 0, 0, 1], dtype=dtype)
        total = cp.full(2, 99, dtype=cp.int64)
        outtags = cp.full(12, 99, dtype=cp.int8)
        outindex = cp.full(12, 99, dtype=cp.int64)
        outoffsets = cp.full(len(tags) + 2, 99, dtype=cp.int64)
        backend = CupyBackend.instance()
        assert (
            backend[
                "awkward_UnionArray_flatten_length", np.int64, np.int8, dtype, np.int64
            ](total, tags, index, len(tags), pointers)
            is None
        )
        assert (
            backend[
                "awkward_UnionArray_flatten_combine",
                np.int8,
                np.int64,
                np.int64,
                np.int8,
                dtype,
                np.int64,
            ](outtags, outindex, outoffsets, tags, index, len(tags), pointers)
            is None
        )
        stream.synchronize()
    assert total.get().tolist() == [0 if empty else 7, 99]
    assert outtags.get().tolist() == (
        [99] * 12 if empty else [1, 1, 1, 0, 0, 1, 1] + [99] * 5
    )
    assert outindex.get().tolist() == (
        [99] * 12 if empty else [3, 4, 5, 2, 3, 1, 2] + [99] * 5
    )
    assert outoffsets.get().tolist() == ([0, 99] if empty else [0, 3, 5, 7, 7, 99])


@pytest.mark.parametrize(
    "tag_dtype,index_dtype",
    [
        (np.int64, np.int64),
        (np.int8, np.int32),
        (np.int8, np.int64),
        (np.int8, np.uint32),
    ],
)
@pytest.mark.parametrize(
    "starts,counts",
    [
        ([], []),
        ([0, 4, 8], [2, 0, 3]),
        ([5, 0, 2], [3, 4, 2]),
        ([0, 1, 2], [4, 3, 2]),
        ([4, 4, 7], [-2, 0, 2]),
    ],
)
def test_nested(tag_dtype, index_dtype, starts, counts):
    expected_tags = np.full(20, 99, dtype=tag_dtype)
    expected_index = np.full(20, 99, dtype=index_dtype)
    k = 0
    for start, count in zip(starts, counts, strict=False):
        for j in range(start, start + count):
            expected_tags[j], expected_index[j] = 3, k
            k += 1
    with cp.cuda.Stream(non_blocking=True) as stream:
        tags, index = (
            cp.asarray(expected_tags * 0 + 99),
            cp.asarray(expected_index * 0 + 99),
        )
        scratch = cp.asarray([*starts, 99], dtype=cp.int64)
        counts_device = cp.asarray(counts, dtype=cp.int64)
        assert (
            CupyBackend.instance()[
                "awkward_UnionArray_nestedfill_tags_index",
                tag_dtype,
                index_dtype,
                np.int64,
                np.int64,
            ](tags, index, scratch, 3, counts_device, len(starts))
            is None
        )
        stream.synchronize()
    np.testing.assert_array_equal(tags.get(), expected_tags)
    np.testing.assert_array_equal(index.get(), expected_index)
    assert scratch.get().tolist() == [
        x + y for x, y in zip(starts, counts, strict=False)
    ] + [99]


def test_public_flatten():
    source = ak.Array([[1, 2], ["x"], [], [3], ["y", "z"]])
    gpu = ak.to_backend(source, "cuda")
    assert ak.to_list(ak.flatten(gpu)) == ak.to_list(ak.flatten(source))


def test_prepared_flatten_rebinds_buffers():
    from awkward._connect.cuda import _unionarray

    previous = []
    with cp.cuda.Stream(non_blocking=True):
        for offsets in ([2, 4, 4], [1, 2, 5]):
            device_offsets = cp.asarray(offsets, dtype=cp.int64)
            pointers = cp.asarray([device_offsets.data.ptr], dtype=cp.int64)
            tags = cp.asarray([0, 0], dtype=cp.int8)
            index = cp.asarray([1, 0], dtype=cp.int64)
            total = cp.full(2, 99, dtype=cp.int64)
            outtags = cp.full(10, 99, dtype=cp.int8)
            outindex = cp.full(10, 99, dtype=cp.int64)
            outoffsets = cp.full(4, 99, dtype=cp.int64)
            _unionarray.flatten_length(total, tags, index, 2, pointers)
            _unionarray.flatten_combine(
                outtags, outindex, outoffsets, tags, index, 2, pointers
            )
            expected = list(range(offsets[1], offsets[2])) + list(
                range(offsets[0], offsets[1])
            )
            assert total.get().tolist() == [len(expected), 99]
            assert outindex.get().tolist() == expected + [99] * (10 - len(expected))
            assert outoffsets.get().tolist() == [
                0,
                offsets[2] - offsets[1],
                len(expected),
                99,
            ]
            previous.append((outindex, expected + [99] * (10 - len(expected))))
        for output, expected in previous:
            assert output.get().tolist() == expected
