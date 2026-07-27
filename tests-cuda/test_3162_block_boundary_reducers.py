from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    try:
        cp.cuda.Device().synchronize()  # wait for all kernels
    except cp.cuda.runtime.CUDARuntimeError as e:
        print("GPU error during sync:", e)
    cp._default_memory_pool.free_all_blocks()


def test_block_boundary_sum():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_any():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    res_flat_cuda = ak.any(cuda_content, -1, highlevel=False)
    res_flat_cpu = ak.any(content, -1, highlevel=False)

    assert bool(res_flat_cuda) == bool(res_flat_cpu)

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    res_seg_cuda = ak.any(cuda_depth1, -1, highlevel=False)
    res_seg_cpu = ak.any(depth1, -1, highlevel=False)

    np.testing.assert_array_equal(ak.to_numpy(res_seg_cuda), ak.to_numpy(res_seg_cpu))

    del cuda_content, cuda_depth1
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_all():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)

    res_flat_cuda = ak.all(cuda_content, -1, highlevel=False)
    res_flat_cpu = ak.all(content, -1, highlevel=False)

    assert bool(res_flat_cuda) == bool(res_flat_cpu)

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    res_seg_cuda = ak.all(cuda_depth1, -1, highlevel=False)
    res_seg_cpu = ak.all(depth1, -1, highlevel=False)

    np.testing.assert_array_equal(ak.to_numpy(res_seg_cuda), ak.to_numpy(res_seg_cpu))

    del cuda_content, cuda_depth1
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_sum_bool():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(2, size=3000, dtype=np.bool_)
    content = ak.contents.NumpyArray(array)
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_max():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    print(array)
    content = ak.contents.NumpyArray(array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    print(ak.max(content, -1, highlevel=False))
    print(ak.max(cuda_content, -1, highlevel=False))
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_min():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_negative_min():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000) * -1
    content = ak.contents.NumpyArray(array)
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_argmin():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_argmax():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_count():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.count(cuda_content, -1, highlevel=False) == ak.count(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.count(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.count(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_count_nonzero():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(2, size=3000)
    content = ak.contents.NumpyArray(array)
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
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_prod():
    # The product of thousands of primes overflows int64 by tens of thousands
    # of bits, which turns this into a test of modular wraparound rather than
    # of `prod`. Instead, build an array that is mostly 1s with a few distinct
    # small prime factors placed exactly on the 1024/2048 thread-block
    # boundaries. The products then stay tiny and exact (so `==` is the right
    # assertion), while still stressing reductions that span block boundaries:
    # if a factor on a boundary were dropped or double-counted, the product
    # would change detectably.
    array = np.ones(3000, dtype=np.int64)
    factors = {0: 2, 1023: 3, 1024: 5, 2047: 7, 2048: 11, 2999: 13}
    for position, value in factors.items():
        array[position] = value
    content = ak.contents.NumpyArray(array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)

    res_flat_cuda = ak.prod(cuda_content, -1, highlevel=False)
    res_flat_cpu = ak.prod(content, -1, highlevel=False)

    # 2 * 3 * 5 * 7 * 11 * 13 == 30030, well within int64.
    assert int(res_flat_cuda) == int(res_flat_cpu) == 30030

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    res_seg_cuda = ak.prod(cuda_depth1, -1, highlevel=False)
    res_seg_cpu = ak.prod(depth1, -1, highlevel=False)

    # bin 0 = [0, 1)       -> 2
    # bin 1 = [1, 2998)    -> 3 * 5 * 7 * 11 = 1155 (spans both block boundaries)
    # bin 2 = [2998, 3000) -> 13
    assert to_list(res_seg_cuda) == to_list(res_seg_cpu) == [2, 1155, 13]

    del cuda_content, cuda_depth1
    cp.get_default_memory_pool().free_all_blocks()


def test_block_boundary_prod_bool():
    # A random bool array almost always contains a False, so its product is
    # ~always 0 and the test degenerates to 0 == 0. Instead make it
    # deterministic and boundary-sensitive: start all-True and drop a single
    # False exactly on a thread-block boundary. Only the bin containing it may
    # collapse to 0, so a kernel that dropped or misplaced the boundary element
    # would change the result.
    array = np.ones(3000, dtype=bool)
    array[2048] = False  # first element of the third block, inside bin 1
    content = ak.contents.NumpyArray(array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)

    res_flat_cuda = ak.prod(cuda_content, -1, highlevel=False)
    res_flat_cpu = ak.prod(content, -1, highlevel=False)

    # A single False makes the whole product False.
    assert int(res_flat_cuda) == int(res_flat_cpu) == 0

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)

    res_seg_cuda = ak.prod(cuda_depth1, -1, highlevel=False)
    res_seg_cpu = ak.prod(depth1, -1, highlevel=False)

    # bin 0 = [0, 1)       -> True
    # bin 1 = [1, 2998)    -> False (contains the boundary False at index 2048)
    # bin 2 = [2998, 3000) -> True
    np.testing.assert_array_equal(ak.to_numpy(res_seg_cuda), ak.to_numpy(res_seg_cpu))
    assert to_list(res_seg_cpu) == [1, 0, 1]

    del cuda_content, cuda_depth1
    cp.get_default_memory_pool().free_all_blocks()
