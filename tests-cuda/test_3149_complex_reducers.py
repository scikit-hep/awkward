from __future__ import annotations

import cupy as cp
import cupy.testing as cpt
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
    assert ak.min(content_complex64, highlevel=False) == (0.25 + 0j)
    assert ak.max(content_complex64, highlevel=False) == (5.5 + 0j)
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


def test_0395_mask_complex():
    content = ak.contents.NumpyArray(
        np.array([(1.1 + 0.1j), 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6, 6, 6, 9], dtype=np.int64))
    array = ak.contents.ListOffsetArray(offsets, content)
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert to_list(ak.min(array, axis=-1, mask_identity=False, highlevel=False)) == [
        (1.1 + 0.1j),
        (np.inf + 0j),
        (4.4 + 0j),
        (6.6 + 0j),
        (np.inf + 0j),
        (np.inf + 0j),
        (7.7 + 0j),
    ]
    assert to_list(ak.min(array, axis=-1, mask_identity=True, highlevel=False)) == [
        (1.1 + 0.1j),
        None,
        (4.4 + 0j),
        (6.6 + 0j),
        None,
        None,
        (7.7 + 0j),
    ]
    del array


def test_0395_count_min_complex():
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

    assert to_list(ak.min(depth1, -1, highlevel=False)) == [(1.1 + 0.1j), 0j, 0j]
    assert to_list(ak.min(depth1, 1, highlevel=False)) == [(1.1 + 0.1j), 0j, 0j]

    content2 = ak.contents.NumpyArray(
        np.array([True, True, True, False, True, False, False, True, False, True])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [True, True, True],
        [False, True, False],
        [False, True, False, True],
    ]

    assert to_list(ak.min(depth1, -1, highlevel=False)) == [True, False, False]
    assert to_list(ak.min(depth1, 1, highlevel=False)) == [True, False, False]
    del depth1


def test_0395_count_max_complex():
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

    assert to_list(ak.max(depth1, -1, highlevel=False)) == [3.3, 2.2, 4.4]
    assert to_list(ak.max(depth1, 1, highlevel=False)) == [3.3, 2.2, 4.4]

    content2 = ak.contents.NumpyArray(
        np.array([False, True, True, False, True, False, False, False, False, False])
    )
    offsets3 = ak.index.Index64(np.array([0, 3, 6, 10], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets3, content2)
    assert to_list(depth1) == [
        [False, True, True],
        [False, True, False],
        [False, False, False, False],
    ]

    assert to_list(ak.max(depth1, -1, highlevel=False)) == [True, True, False]
    assert to_list(ak.max(depth1, 1, highlevel=False)) == [True, True, False]
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


def test_0652_reducers():
    array = ak.operations.from_iter([[1 + 1j, 2 + 2j], [], [3 + 3j]])
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert ak.operations.sum(array) == 6 + 6j
    assert ak.operations.prod(array) == -12 + 12j

    assert ak.operations.sum(array, axis=1).to_list() == [
        3 + 3j,
        0 + 0j,
        3 + 3j,
    ]
    assert ak.operations.prod(array, axis=1).to_list() == [
        0 + 4j,
        1 + 0j,
        3 + 3j,
    ]

    assert ak.operations.count(array, axis=1).to_list() == [2, 0, 1]
    assert ak.operations.count_nonzero(array, axis=1).to_list() == [2, 0, 1]
    assert ak.operations.any(array, axis=1).to_list() == [True, False, True]
    assert ak.operations.all(array, axis=1).to_list() == [True, True, True]

    array = ak.operations.from_iter([[1 + 1j, 2 + 2j, 0 + 0j], [], [3 + 3j]])
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert ak.operations.any(array, axis=1).to_list() == [True, False, True]
    assert ak.operations.all(array, axis=1).to_list() == [False, True, True]
    del array


def test_0652_minmax():
    array = ak.operations.from_iter([[1 + 5j, 2 + 4j], [], [3 + 3j]])
    array = ak.to_backend(array, "cuda", highlevel=False)

    assert ak.operations.min(array) == 1 + 5j
    assert ak.operations.max(array) == 3 + 3j

    assert ak.operations.min(array, axis=1).to_list() == [
        1 + 5j,
        None,
        3 + 3j,
    ]
    assert ak.operations.max(array, axis=1).to_list() == [
        2 + 4j,
        None,
        3 + 3j,
    ]
    del array


def test_block_boundary_sum_complex():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
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


def test_block_boundary_prod_complex1():
    complex_array = np.vectorize(complex)(np.full(1000, 0), np.full(1000, 1))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.prod(cuda_content, -1, highlevel=False) == ak.prod(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.prod(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.prod(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex2():
    complex_array = np.vectorize(complex)(np.full(1001, 0), np.full(1001, 1))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.prod(cuda_content, -1, highlevel=False) == ak.prod(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1001], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.prod(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.prod(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex3():
    complex_array = np.vectorize(complex)(np.full(1002, 0), np.full(1002, 1))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.prod(cuda_content, -1, highlevel=False) == ak.prod(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 5, 999, 1002], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.prod(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.prod(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex4():
    complex_array = np.vectorize(complex)(np.full(1000, 0), np.full(1000, 1.01))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex5():
    complex_array = np.vectorize(complex)(np.full(1001, 0), np.full(1001, 1.01))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1001], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex6():
    complex_array = np.vectorize(complex)(np.full(1002, 0), np.full(1002, 1.01))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 998, 1002], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex7():
    complex_array = np.vectorize(complex)(np.full(1000, 0), np.full(1000, 0.99))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex8():
    complex_array = np.vectorize(complex)(np.full(1001, 0), np.full(1001, 0.99))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1001], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex9():
    complex_array = np.vectorize(complex)(np.full(1002, 0), np.full(1002, 0.99))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 999, 1002], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex10():
    complex_array = np.vectorize(complex)(np.full(1000, 0), np.full(1000, 1.1))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex11():
    complex_array = np.vectorize(complex)(np.full(1001, 0), np.full(1001, 1.1))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1001], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex12():
    complex_array = np.vectorize(complex)(np.full(1002, 0), np.full(1002, 1.1))
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1002], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_complex13():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(50, size=1000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    cpt.assert_allclose(
        ak.prod(cuda_content, -1, highlevel=False),
        ak.prod(content, -1, highlevel=False),
    )

    offsets = ak.index.Index64(np.array([0, 5, 996, 1000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    cpt.assert_allclose(
        to_list(ak.prod(cuda_depth1, -1, highlevel=False)),
        to_list(ak.prod(depth1, -1, highlevel=False)),
    )
    del cuda_content, cuda_depth1


def test_block_boundary_any_complex():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.any(cuda_content, -1, highlevel=False) == ak.any(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.any(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.any(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_all_complex():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.all(cuda_content, -1, highlevel=False) == ak.all(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.all(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.all(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_min_complex1():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(5, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.min(cuda_content, -1, highlevel=False) == ak.min(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.min(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.min(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_min_complex2():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.min(cuda_content, -1, highlevel=False) == ak.min(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.min(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.min(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_max_complex1():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(5, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.max(cuda_content, -1, highlevel=False) == ak.max(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.max(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.max(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_max_complex2():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.max(cuda_content, -1, highlevel=False) == ak.max(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.max(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.max(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_sum_bool_complex():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(2, size=6000, dtype=np.bool_)
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
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
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
    rng = np.random.default_rng(seed=42)
    array = rng.integers(2, size=6000)
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


@pytest.mark.skip(reason="awkward_reduce_argmax_complex is not implemented")
def test_block_boundary_argmax_complex1():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(5, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.argmax(cuda_content, -1, highlevel=False) == ak.argmax(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.argmax(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.argmax(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


@pytest.mark.skip(reason="awkward_reduce_argmax_complex is not implemented")
def test_block_boundary_argmax_complex2():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.argmax(cuda_content, -1, highlevel=False) == ak.argmax(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.argmax(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.argmax(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


@pytest.mark.skip(reason="awkward_reduce_argmin_complex is not implemented")
def test_block_boundary_argmin_complex1():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(5, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.argmin(cuda_content, -1, highlevel=False) == ak.argmin(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.argmin(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.argmin(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


@pytest.mark.skip(reason="awkward_reduce_argmin_complex is not implemented")
def test_block_boundary_argmin_complex2():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(6000, size=6000)
    complex_array = np.vectorize(complex)(
        array[0 : len(array) : 2], array[1 : len(array) : 2]
    )
    content = ak.contents.NumpyArray(complex_array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.argmin(cuda_content, -1, highlevel=False) == ak.argmin(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.argmin(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.argmin(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1
