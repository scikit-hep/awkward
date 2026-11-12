import math

import cupy as cp
import cupy.testing as cpt
import numpy as np
import pytest

import awkward as ak


@pytest.fixture(scope="function", autouse=True)
def cleanup_cuda():
    yield
    # Surface asynchronous failures without discarding reusable pool allocations.
    cp.cuda.Device().synchronize()


@pytest.fixture(
    scope="module",
    params=[
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ],
    ids=lambda dtype: np.dtype(dtype).name,
)
def reducer_inputs(request):
    dtype = request.param
    values = (
        [[True, False, False], [True, False, False]]
        if dtype is np.bool_
        else [[0, 1, 2], [3, 4, 5]]
    )
    array = np.array(values, dtype=dtype)
    content = ak.contents.NumpyArray(array.reshape(-1))
    offsets = ak.index.Index64(np.array([0, 3, 3, 5, 6], dtype=np.int64))
    depth = ak.contents.ListOffsetArray(offsets, content)
    return array, ak.to_backend(depth, "cuda")


def test_0115_generic_reducer_operation_sumprod_types(reducer_inputs):
    array, depth = reducer_inputs
    sum_result = ak.to_cupy(ak.sum(depth, axis=-1, highlevel=False))
    prod_result = ak.to_cupy(ak.prod(depth, axis=-1, highlevel=False))

    assert sum_result.dtype == np.sum(array, axis=-1).dtype
    assert prod_result.dtype == np.prod(array, axis=-1).dtype
    assert sum(ak.to_list(np.sum(array, axis=-1))) == sum(sum_result.tolist())
    assert math.prod(ak.to_list(np.prod(array, axis=-1))) == math.prod(
        prod_result.tolist()
    )

    # Check each ragged segment, including the empty-list identities, against NumPy.
    flat = array.reshape(-1)
    segments = [flat[:3], flat[3:3], flat[3:5], flat[5:6]]
    cpt.assert_array_equal(sum_result, cp.asarray([np.sum(x) for x in segments]))
    cpt.assert_array_equal(prod_result, cp.asarray([np.prod(x) for x in segments]))


@pytest.fixture(scope="module")
def cuda_array():
    return ak.Array(
        [[0, 2, 3.0], [4, 5, 6, 7, 8], [], [9, 8, None], [10, 1], []],
        backend="cuda",
    )


@pytest.fixture(scope="module")
def mask_templates():
    return {
        True: ak.Array([[True]], backend="cuda"),
        False: ak.Array([[False]], backend="cuda"),
    }


def test_2020_reduce_axis_none_sum(cuda_array, mask_templates):
    array = cuda_array
    cpt.assert_allclose(ak.sum(array, axis=None), 63.0)
    assert ak.almost_equal(
        ak.sum(array, axis=None, keepdims=True),
        ak.to_regular(ak.Array([[63.0]], backend="cuda")),
    )

    arr = ak.Array([[63.0]], backend="cuda")
    assert ak.almost_equal(
        ak.sum(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[True]]),
    )
    assert ak.sum(array[2], axis=None, mask_identity=True) is None


def test_2020_reduce_axis_none_prod(cuda_array, mask_templates):
    array = cuda_array
    cpt.assert_allclose(ak.prod(array[1:], axis=None), 4838400.0)
    assert ak.prod(array, axis=None) == 0
    assert ak.almost_equal(
        ak.prod(array, axis=None, keepdims=True),
        ak.to_regular(ak.Array([[0.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.prod(array[1:], axis=None, keepdims=True),
        ak.to_regular(ak.Array([[4838400.0]], backend="cuda")),
    )

    arr = ak.Array([[4838400.0]], backend="cuda")
    assert ak.almost_equal(
        ak.prod(array[1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[True]]),
    )
    assert ak.prod(array[2], axis=None, mask_identity=True) is None


def test_2020_reduce_axis_none_min(cuda_array, mask_templates):
    array = cuda_array
    cpt.assert_allclose(ak.min(array, axis=None), 0.0)
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[0.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, initial=-100.0, mask_identity=False),
        ak.to_regular(ak.Array([[-100.0]], backend="cuda")),
    )

    arr = ak.Array([[0.0]], backend="cuda")
    assert ak.almost_equal(
        ak.min(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[True]]),
    )

    arr = ak.Array(ak.Array([[np.inf]], backend="cuda"))
    assert ak.almost_equal(
        ak.min(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[False]]),
    )
    assert ak.min(array[2], axis=None, mask_identity=True) is None


def test_2020_reduce_axis_none_max(cuda_array, mask_templates):
    array = cuda_array
    cpt.assert_allclose(ak.max(array, axis=None), 10.0)
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[10.0]], backend="cuda")),
    )
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, initial=100.0, mask_identity=False),
        ak.to_regular(ak.Array([[100.0]], backend="cuda")),
    )

    arr = ak.Array([[10.0]], backend="cuda")
    assert ak.almost_equal(
        ak.max(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[True]]),
    )

    arr = ak.Array(ak.Array([[np.inf]], backend="cuda"))
    assert ak.almost_equal(
        ak.max(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[False]]),
    )
    assert ak.max(array[2], axis=None, mask_identity=True) is None


def test_2020_reduce_axis_none_count(cuda_array, mask_templates):
    array = cuda_array
    assert ak.count(array, axis=None) == 12
    assert ak.almost_equal(
        ak.count(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[12]], backend="cuda")),
    )

    arr = ak.Array([[12]], backend="cuda")
    assert ak.almost_equal(
        ak.count(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[True]]),
    )

    arr = ak.Array([[0]], backend="cuda")
    assert ak.almost_equal(
        ak.count(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[False]]),
    )
    assert ak.count(array[2], axis=None, mask_identity=True) is None
    assert ak.count(array[2], axis=None, mask_identity=False) == 0


def test_2020_reduce_axis_none_count_nonzero(cuda_array, mask_templates):
    array = cuda_array
    assert ak.count_nonzero(array, axis=None) == 11
    assert ak.almost_equal(
        ak.count_nonzero(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[11]], backend="cuda")),
    )

    arr = ak.Array([[11]], backend="cuda")
    assert ak.almost_equal(
        ak.count_nonzero(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[True]]),
    )

    arr = ak.Array([[0]], backend="cuda")
    assert ak.almost_equal(
        ak.count_nonzero(array[-1:], axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[False]]),
    )
    assert ak.count_nonzero(array[2], axis=None, mask_identity=True) is None
    assert ak.count_nonzero(array[2], axis=None, mask_identity=False) == 0


def test_2020_reduce_axis_none_std_no_mask_axis_none(cuda_array, mask_templates):
    array = cuda_array
    out1 = ak.std(array[-1:], axis=None, keepdims=True, mask_identity=True)

    arr = ak.Array([[0.0]], backend="cuda")
    out2 = ak.to_regular(arr.mask[mask_templates[False]])
    assert ak.almost_equal(out1, out2)

    out3 = ak.std(array[2], axis=None, mask_identity=True)
    assert out3 is None


def test_2020_reduce_axis_none_std(cuda_array, mask_templates):
    array = cuda_array
    cpt.assert_allclose(ak.std(array, axis=None), 3.139134700306227)
    assert ak.almost_equal(
        ak.std(array, axis=None, keepdims=True, mask_identity=False),
        ak.to_regular(ak.Array([[3.139134700306227]], backend="cuda")),
    )

    arr = ak.Array([[3.139134700306227]], backend="cuda")
    assert ak.almost_equal(
        ak.std(array, axis=None, keepdims=True, mask_identity=True),
        ak.to_regular(arr.mask[mask_templates[True]]),
    )
    assert np.isnan(ak.std(array[2], axis=None, mask_identity=False))
