# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytest.importorskip("cupy")
pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
@pytest.mark.parametrize("k", [0, 1, 4])
def test_ragged(axis, mode, k):
    cpu = ak.Array([[3.0, 1.0], [], [2.0, None, 4.0]])
    gpu = ak.to_backend(cpu, "cuda")
    expected = ak.top_k(cpu, k, axis=axis, mode=mode)
    result = ak.top_k(gpu, k, axis=axis, mode=mode)
    for actual, reference in zip(result, expected, strict=True):
        assert ak.backend(actual) == "cuda"
        assert ak.to_list(actual) == ak.to_list(reference)
        assert actual.layout.form == reference.layout.form
    assert ak.to_list(ak.top_k_values(gpu, k, axis=axis, mode=mode)) == ak.to_list(
        result.values
    )
    assert ak.to_list(ak.top_k_indices(gpu, k, axis=axis, mode=mode)) == ak.to_list(
        result.indices
    )
    ak._connect.cuda.synchronize_cuda()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_nested(axis):
    cpu = ak.Array([[[1.0, 5.0], None], [[2.0], [3.0, 4.0]]])
    result = ak.top_k(ak.to_backend(cpu, "cuda"), 2, axis=axis)
    expected = ak.top_k(cpu, 2, axis=axis)
    for actual, reference in zip(result, expected, strict=True):
        assert ak.to_list(actual) == ak.to_list(reference)
        assert actual.layout.form == reference.layout.form
    ak._connect.cuda.synchronize_cuda()


def test_rectangular():
    array = np.array([[3, 1, 2], [5, 4, 6]], dtype=np.int64)
    result = ak.top_k(ak.to_backend(array, "cuda"), 1, axis=0)
    assert ak.to_list(result.values) == [[5, 4, 6]]
    assert ak.to_list(result.indices) == [[1, 1, 1]]
    ak._connect.cuda.synchronize_cuda()


@pytest.mark.parametrize(
    "counts, outer_offsets, maxcount, prefix",
    [
        ([3, 1, 0, 2], [0, 3, 4], 4, 0),
        ([2, 1], [0, 2, 2], 3, 2),
        ([0, 0], [0, 1, 2], 3, 0),
        ([], [0], 3, 0),
        ([], [0, 0], 0, 0),
        ([1, 0, 513], [0, 3], 513, 0),
        ([i % 4 for i in range(600)], [0, 200, 400, 600], 4, 0),
    ],
)
def test_nonlocal_nextshifts_matches_cpu(counts, outer_offsets, maxcount, prefix):
    import cupy as cp

    from awkward._backends.cupy import CupyBackend
    from awkward._backends.numpy import NumpyBackend

    offsets = np.concatenate(([prefix], prefix + np.cumsum(counts))).astype(np.int64)
    # Reordered and repeated gathers must read the completed missing buffer.
    carry = np.arange(prefix, offsets[-1], dtype=np.int64)[::-1].copy()
    carry = np.concatenate((carry, carry[:2]))
    outer = np.asarray(outer_offsets, dtype=np.int64)
    arrays = [
        np.full(maxcount, -123, dtype=np.int64),
        np.full(offsets[-1], -123, dtype=np.int64),
        np.full(len(carry), -123, dtype=np.int64),
        offsets,
        len(counts),
        outer[:-1].copy(),
        outer,
        len(outer) - 1,
        maxcount,
        len(carry),
        carry,
    ]
    gpu_args = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in arrays]
    key = ("awkward_ListOffsetArray_reduce_nonlocal_nextshifts_64",) + (np.int64,) * 7
    error = NumpyBackend.instance()[key](*arrays)
    assert not error.str
    CupyBackend.instance()[key](*gpu_args)
    ak._connect.cuda.synchronize_cuda()
    for expected, actual in zip(arrays[:3], gpu_args[:3], strict=True):
        np.testing.assert_array_equal(cp.asnumpy(actual), expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_deep_alignment_keeps_cuda_backend(axis):
    # Exercise both alignment and trimming with more than one inner dimension.
    cpu = ak.Array([[[[3], [1, 5]], None], [[[2, 4]], [[6], []]]])
    gpu = ak.to_backend(cpu, "cuda")
    expected = ak.top_k(cpu, 2, axis=axis, highlevel=False)
    result = ak.top_k(gpu, 2, axis=axis, highlevel=False)
    for actual, reference in zip(result, expected, strict=True):
        assert actual.backend is gpu.layout.backend
        assert actual.form == reference.form
        assert ak.to_list(actual) == ak.to_list(reference)
    ak._connect.cuda.synchronize_cuda()
