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


def test_0395_count_complex():
    content2 = ak.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [(1.1 + 0.1j), (2.2 + 0j), (3.3 + 0j)],
        [0j, (2.2 + 0j), 0j],
        [0j, (2.2 + 0j), 0j, (4.4 + 0j)],
    ]

    assert to_list(ak.count(depth1, -1, highlevel=False)) == [3, 3, 4]
    assert to_list(ak.count(depth1, 1, highlevel=False)) == [3, 3, 4]

    del depth1


def test_0395_count_nonzero_complex():
    content2 = ak.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 0.0, 2.2, 0.0, 0.0, 2.2, 0.0, 4.4])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(depth1) == [
        [(1.1 + 0.1j), (2.2 + 0j), (3.3 + 0j)],
        [0j, (2.2 + 0j), 0j],
        [0j, (2.2 + 0j), 0j, (4.4 + 0j)],
    ]

    assert to_list(ak.count_nonzero(depth1, -1, highlevel=False)) == [3, 1, 2]
    assert to_list(ak.count_nonzero(depth1, 1, highlevel=False)) == [3, 1, 2]

    del depth1


def test_reducers():
    array = ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]])
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert ak.operations.sum(array) == 6 + 6j
    # assert ak.operations.prod(array) == -12 + 12j

    assert ak.operations.sum(array, axis=1).to_list() == [
        3 + 3j,
        0 + 0j,
        3 + 3j,
    ]
    # assert ak.operations.prod(array, axis=1).to_list() == [
    #     0 + 4j,
    #     1 + 0j,
    #     3 + 3j,
    # ]

    assert ak.operations.count(array, axis=1).to_list() == [2, 0, 1]
    assert ak.operations.count_nonzero(array, axis=1).to_list() == [2, 0, 1]
    assert ak.operations.any(array, axis=1).to_list() == [True, False, True]
    assert ak.operations.all(array, axis=1).to_list() == [True, True, True]

    array = ak.operations.from_iter([[1 + 1j, 2 + 2j, 0 + 0j], [], [3 + 3j]])
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert ak.operations.any(array, axis=1).to_list() == [True, False, True]
    assert ak.operations.all(array, axis=1).to_list() == [False, True, True]


def test_block_boundary_sum_complex():
    np.random.seed(42)
    array = np.random.randint(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.sum(cuda_content, -1, highlevel=False) == ak.sum(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.sum(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.sum(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_countnonzero_complex_1():
    np.random.seed(42)
    array = np.random.randint(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.count_nonzero(cuda_content, -1, highlevel=False) == ak.count_nonzero(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(ak.count_nonzero(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.count_nonzero(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_countnonzero_complex_2():
    np.random.seed(42)
    array = np.random.randint(2, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.count_nonzero(cuda_content, -1, highlevel=False) == ak.count_nonzero(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    assert to_list(ak.count_nonzero(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.count_nonzero(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1
