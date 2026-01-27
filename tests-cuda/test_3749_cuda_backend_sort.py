# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_sort_cuda_basic():
    data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
    gpu_data = ak.to_backend(data, "cuda")
    gpu_sorted = ak.sort(gpu_data)
    result = ak.to_backend(gpu_sorted, "cpu")

    assert to_list(result) == [[5, 7, 7], [], [2], [2, 8]]


def test_sort_cuda_descending():
    data = ak.Array([[3, 1, 2], [5, 4], [], [6]])
    gpu_data = ak.to_backend(data, "cuda")
    gpu_sorted = ak.sort(gpu_data, ascending=False)
    result = ak.to_backend(gpu_sorted, "cpu")

    assert to_list(result) == [[3, 2, 1], [5, 4], [], [6]]


def test_sort_cuda_float():
    data = ak.Array([[3.5, 1.2, 2.8], [5.1, 4.9], [], [6.0]])
    gpu_data = ak.to_backend(data, "cuda")
    gpu_sorted = ak.sort(gpu_data)
    result = ak.to_backend(gpu_sorted, "cpu")

    expected = [[1.2, 2.8, 3.5], [4.9, 5.1], [], [6.0]]
    result_list = to_list(result)

    # Compare with tolerance for floats
    assert len(result_list) == len(expected)
    for res_sublist, exp_sublist in zip(result_list, expected, strict=True):
        assert len(res_sublist) == len(exp_sublist)
        for res_val, exp_val in zip(res_sublist, exp_sublist, strict=True):
            assert abs(res_val - exp_val) < 1e-10


def test_sort_cuda_large():
    # Create random data
    np.random.seed(42)
    data_list = []
    for _ in range(100):
        size = np.random.randint(0, 50)
        if size > 0:
            data_list.append(np.random.randint(0, 100, size).tolist())
        else:
            data_list.append([])

    data = ak.Array(data_list)
    cpu_sorted = ak.sort(data)

    gpu_data = ak.to_backend(data, "cuda")
    gpu_sorted = ak.sort(gpu_data)
    result = ak.to_backend(gpu_sorted, "cpu")

    assert to_list(result) == to_list(cpu_sorted)


def test_sort_cuda_nested():
    data = ak.Array([[[3, 1, 2], [5, 4]], [[9, 7, 8]], [[6]]])
    cpu_sorted = ak.sort(data, axis=-1)

    gpu_data = ak.to_backend(data, "cuda")
    gpu_sorted = ak.sort(gpu_data, axis=-1)
    result = ak.to_backend(gpu_sorted, "cpu")

    assert to_list(result) == to_list(cpu_sorted)
    assert to_list(result) == [[[1, 2, 3], [4, 5]], [[7, 8, 9]], [[6]]]


def test_sort_cuda_deeply_nested():
    data = ak.Array([[[[5, 2, 8], [1, 3]], [[4, 6]]], [[[9, 7]]]])
    cpu_sorted = ak.sort(data, axis=-1)

    gpu_data = ak.to_backend(data, "cuda")
    gpu_sorted = ak.sort(gpu_data, axis=-1)
    result = ak.to_backend(gpu_sorted, "cpu")

    assert to_list(result) == to_list(cpu_sorted)
    assert to_list(result) == [[[[2, 5, 8], [1, 3]], [[4, 6]]], [[[7, 9]]]]


def test_sort_cuda_unsupported_axis():
    """Test that sorting at unsupported axes fails with clear error."""
    # Sorting at axis=-2 requires CuPy kernels that don't exist
    # This should fail with an AssertionError indicating missing kernels
    data = ak.Array([[[7, 2, 3], [4, 5, 6]]])
    gpu_data = ak.to_backend(data, "cuda")

    # axis=-1 should work (our cuda.compute implementation)
    sorted_axis_minus1 = ak.sort(gpu_data, axis=-1)
    result = ak.to_backend(sorted_axis_minus1, "cpu")
    assert to_list(result) == [[[2, 3, 7], [4, 5, 6]]]

    # axis=-2 should fail (requires CuPy kernels not available)
    with pytest.raises(
        AssertionError,
        match=r"(CuPyKernel not found|Operation .* is not supported)",
    ):
        ak.sort(gpu_data, axis=-2)


def test_sort_cuda_no_compute():
    """Test that helpful error is raised when cuda.compute is not available."""
    from awkward._connect.cuda import _compute as cuda_compute

    original_available = cuda_compute._cuda_compute_available

    try:
        # Temporarily make cuda.compute unavailable
        cuda_compute._cuda_compute_available = False

        data = ak.Array([[7, 5, 7], [], [2], [8, 2]])
        gpu_data = ak.to_backend(data, "cuda")

        with pytest.raises(NotImplementedError, match=r"cuda\.compute"):
            ak.sort(gpu_data)

    finally:
        # Restore original state
        cuda_compute._cuda_compute_available = original_available
