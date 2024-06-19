from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()


def test_0652_tests_of_complex_numbers_reducers():
    array = [[1 + 1j, 2 + 2j], [], [3 + 3j]]
    cuda_array = ak.to_backend(array, "cuda")

    assert ak.operations.sum(ak.operations.from_iter(cuda_array)) == 6 + 6j

    assert ak.operations.sum(ak.operations.from_iter(cuda_array), axis=1).to_list() == [
        3 + 3j,
        0 + 0j,
        3 + 3j,
    ]
    del cuda_array


def test_0395_astype_complex():
    content_float64 = ak.contents.NumpyArray(
        np.array([0.25, 0.5, 3.5, 4.5, 5.5], dtype=np.float64)
    )
    cuda_content_float64 = ak.to_backend(content_float64, "cuda", highlevel=False)

    array_float64 = ak.contents.UnmaskedArray(content_float64)
    cuda_array_float64 = ak.to_backend(array_float64, "cuda", highlevel=False)

    assert to_list(cuda_array_float64) == [0.25, 0.5, 3.5, 4.5, 5.5]
    assert str(ak.operations.type(cuda_content_float64)) == "5 * float64"
    assert (
        str(ak.operations.type(ak.highlevel.Array(cuda_content_float64)))
        == "5 * float64"
    )
    assert str(ak.operations.type(cuda_array_float64)) == "5 * ?float64"
    assert (
        str(ak.operations.type(ak.highlevel.Array(cuda_array_float64)))
        == "5 * ?float64"
    )
    assert str(cuda_content_float64.form.type) == "float64"
    assert str(cuda_array_float64.form.type) == "?float64"

    del cuda_content_float64, cuda_array_float64

    content_complex64 = ak.operations.values_astype(
        content_float64, "complex64", highlevel=False
    )
    cuda_content_complex64 = ak.to_backend(content_complex64, "cuda", highlevel=False)

    array_complex64 = ak.contents.UnmaskedArray(content_complex64)
    cuda_array_complex64 = ak.to_backend(array_complex64, "cuda", highlevel=False)

    assert to_list(cuda_content_complex64) == [
        (0.25 + 0j),
        (0.5 + 0j),
        (3.5 + 0j),
        (4.5 + 0j),
        (5.5 + 0j),
    ]
    assert to_list(cuda_array_complex64) == [
        (0.25 + 0.0j),
        (0.5 + 0.0j),
        (3.5 + 0.0j),
        (4.5 + 0.0j),
        (5.5 + 0.0j),
    ]
    assert str(ak.operations.type(cuda_content_complex64)) == "5 * complex64"
    assert (
        str(ak.operations.type(ak.highlevel.Array(cuda_content_complex64)))
        == "5 * complex64"
    )
    assert str(ak.operations.type(cuda_array_complex64)) == "5 * ?complex64"
    assert (
        str(ak.operations.type(ak.highlevel.Array(cuda_array_complex64)))
        == "5 * ?complex64"
    )
    assert str(cuda_content_complex64.form.type) == "complex64"
    assert str(cuda_array_complex64.form.type) == "?complex64"

    content = ak.contents.NumpyArray(
        np.array([1, (2.2 + 0.1j), 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    cuda_content = ak.to_backend(content, "cuda")

    assert to_list(cuda_content) == [
        (1 + 0j),
        (2.2 + 0.1j),
        (3.3 + 0j),
        (4.4 + 0j),
        (5.5 + 0j),
        (6.6 + 0j),
        (7.7 + 0j),
        (8.8 + 0j),
        (9.9 + 0j),
    ]

    assert ak.sum(cuda_content_complex64) == (14.25 + 0j)
    del cuda_content_complex64, cuda_array_complex64
    del content
