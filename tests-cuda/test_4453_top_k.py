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
