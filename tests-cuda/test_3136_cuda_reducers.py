from __future__ import annotations

import cupy as cp
import cupy.testing as cpt
import numpy as np
import pytest

import awkward as ak


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    cp.cuda.Stream.null.synchronize()


def assert_gpu_equal_with_dtype(result, expected):
    cp.testing.assert_allclose(result, expected)
    assert result.dtype == expected.dtype


@pytest.fixture(scope="module")
def depth(request):
    dtype = request.param
    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype)

    content = ak.contents.NumpyArray(array.reshape(-1))
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth = ak.contents.ListOffsetArray(offsets, content)

    return depth


@pytest.fixture(scope="module")
def depth_gpu(request):
    dtype = request.param
    array = np.array([[0, 1, 2], [3, 4, 5]], dtype=dtype)

    content = ak.contents.NumpyArray(array.reshape(-1))
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth = ak.contents.ListOffsetArray(offsets, content)

    return ak.to_backend(depth, "cuda")


@pytest.mark.parametrize(
    "depth,depth_gpu,dtype",
    [
        (None, None, np.int8),
        (None, None, np.uint8),
        (None, None, np.int16),
        (None, None, np.uint16),
        (None, None, np.int32),
        (None, None, np.uint32),
        (None, None, np.int64),
        (None, None, np.uint64),
    ],
    indirect=["depth", "depth_gpu"],
)
def test_sumprod_gpu(dtype, depth, depth_gpu):
    sum_gpu = ak.sum(depth_gpu, axis=-1, highlevel=False)
    prod_gpu = ak.prod(depth_gpu, axis=-1, highlevel=False)

    sum_expected = cp.asarray(ak.sum(depth, axis=-1, highlevel=False))
    prod_expected = cp.asarray(ak.prod(depth, axis=-1, highlevel=False))

    assert_gpu_equal_with_dtype(sum_gpu, sum_expected)
    assert_gpu_equal_with_dtype(prod_gpu, prod_expected)


def test_0115_generic_reducer_operation_sumprod_types():
    array = np.array([[True, False, False], [True, False, False]])
    content = ak.contents.NumpyArray(array.reshape(-1))
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth = ak.contents.ListOffsetArray(offsets, content)

    depth_gpu = ak.to_backend(depth, "cuda")

    sum_gpu = ak.sum(depth_gpu, axis=-1, highlevel=False)
    prod_gpu = ak.prod(depth_gpu, axis=-1, highlevel=False)

    sum_expected = cp.asarray(ak.sum(depth, axis=-1))
    prod_expected = cp.asarray(ak.prod(depth, axis=-1))

    assert_gpu_equal_with_dtype(sum_gpu, sum_expected)
    assert_gpu_equal_with_dtype(prod_gpu, prod_expected)


@pytest.fixture(scope="module")
def cuda_array():
    return ak.Array(
        [[0, 2, 3.0], [4, 5, 6, 7, 8], [], [9, 8, None], [10, 1], []],
        backend="cuda",
    )


@pytest.fixture(scope="module")
def expected_scalars():
    return {
        "sum": 63.0,
        "prod": 0.0,
        "prod_slice": 4838400.0,
        "min": 0.0,
        "max": 10.0,
        "count": 12,
        "count_nonzero": 11,
        "std": 3.139134700306227,
    }


@pytest.fixture(scope="module")
def mask_templates():
    return {
        "true": ak.Array([[True]], backend="cuda"),
        "false": ak.Array([[False]], backend="cuda"),
    }


def test_reduce_axis_none_all(cuda_array, expected_scalars, mask_templates):
    arr = cuda_array

    # --- SUM ---
    cpt.assert_allclose(ak.sum(arr, axis=None), expected_scalars["sum"])

    out = ak.sum(arr, axis=None, keepdims=True)
    expected = ak.full_like(out, expected_scalars["sum"])
    assert ak.almost_equal(out, expected)

    mask = ak.full_like(out, True, dtype=bool)
    masked = ak.mask(out, mask)

    assert ak.almost_equal(
        ak.sum(arr, axis=None, keepdims=True, mask_identity=True),
        masked,
    )

    assert ak.sum(arr[2], axis=None, mask_identity=True) is None

    # --- PROD ---
    cpt.assert_allclose(ak.prod(arr[1:], axis=None), expected_scalars["prod_slice"])
    assert ak.prod(arr, axis=None) == expected_scalars["prod"]

    # --- MIN / MAX ---
    cpt.assert_allclose(ak.min(arr, axis=None), expected_scalars["min"])
    cpt.assert_allclose(ak.max(arr, axis=None), expected_scalars["max"])

    # --- COUNT ---
    assert ak.count(arr, axis=None) == expected_scalars["count"]
    assert ak.count_nonzero(arr, axis=None) == expected_scalars["count_nonzero"]

    # --- STD ---
    cpt.assert_allclose(ak.std(arr, axis=None), expected_scalars["std"])
