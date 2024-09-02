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


def test_block_boundary_any():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
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


def test_block_boundary_all():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(3000, size=3000)
    content = ak.contents.NumpyArray(array)
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


@pytest.mark.skip(reason="awkward_reduce_argmin is not implemented")
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


@pytest.mark.skip(reason="awkward_reduce_argmax is not implemented")
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


def test_block_boundary_prod():
    primes = [x for x in range(2, 30000) if all(x % n != 0 for n in range(2, x))]
    content = ak.contents.NumpyArray(primes)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.prod(cuda_content, -1, highlevel=False) == ak.prod(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.prod(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.prod(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1


def test_block_boundary_prod_bool():
    rng = np.random.default_rng(seed=42)
    array = rng.integers(2, size=3000, dtype=np.bool_)
    content = ak.contents.NumpyArray(array)
    cuda_content = ak.to_backend(content, "cuda", highlevel=False)
    assert ak.prod(cuda_content, -1, highlevel=False) == ak.prod(
        content, -1, highlevel=False
    )

    offsets = ak.index.Index64(np.array([0, 1, 2998, 3000], dtype=np.int64))
    depth1 = ak.contents.ListOffsetArray(offsets, content)
    cuda_depth1 = ak.to_backend(depth1, "cuda", highlevel=False)
    assert to_list(ak.prod(cuda_depth1, -1, highlevel=False)) == to_list(
        ak.prod(depth1, -1, highlevel=False)
    )
    del cuda_content, cuda_depth1
