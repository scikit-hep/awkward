# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import awkward as ak
from awkward._backends.cupy import CupyBackend
from awkward._backends.numpy import NumpyBackend

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")

spec = importlib.util.spec_from_file_location(
    "listarray_contract_cases",
    Path(__file__).resolve().parents[1] / "tests/test_4481_listarray_compute.py",
)
contract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = contract
spec.loader.exec_module(contract)


@pytest.mark.parametrize("name,specialization,case", contract.CASES)
def test_cuda_cpu_parity(name, specialization, case):
    args, outputs, key = contract.arguments(specialization, case)
    gpu_args = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
    expected = NumpyBackend.instance()[(name, *key)](*args)
    kernel = CupyBackend.instance()[(name, *key)]
    assert isinstance(kernel, ak._kernels.CudaComputeKernel)
    actual = kernel(*gpu_args)
    contract.assert_error(actual, expected)
    for i in outputs:
        np.testing.assert_array_equal(cp.asnumpy(gpu_args[i]), args[i])


@pytest.mark.parametrize("stream", ["default", "nondefault"])
@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("exception", [False, True])
def test_invalid_list_validity(stream, offsets, exception):
    stream = (
        cp.cuda.Stream.null
        if stream == "default"
        else cp.cuda.Stream(non_blocking=True)
    )
    with stream:
        content = ak.contents.NumpyArray(cp.asarray([1]))
        if offsets:
            layout = ak.contents.ListOffsetArray(
                ak.index.Index64(cp.asarray([0, 2], dtype=cp.int64)), content
            )
        else:
            layout = ak.contents.ListArray(
                ak.index.Index64(cp.asarray([0], dtype=cp.int64)),
                ak.index.Index64(cp.asarray([2], dtype=cp.int64)),
                content,
            )
        if exception:
            with pytest.raises(ValueError, match="stop"):
                ak.validity_error(layout, exception=True)
            with pytest.raises(ValueError, match="stop"):
                ak.is_valid(layout, exception=True)
        else:
            assert "stop" in ak.validity_error(layout)
            assert ak.is_valid(layout) is False
        ak._connect.cuda.synchronize_cuda(stream)
        assert ak.is_valid(ak.to_backend([1, 2], "cuda"))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_highlevel_listarray_operations(dtype):
    with cp.cuda.Stream(non_blocking=True):
        _highlevel_listarray_operations(dtype)


def _highlevel_listarray_operations(dtype):
    source = ak.Array(
        ak.contents.ListArray(
            ak.index.Index(np.asarray([0, 4, 2, 7], dtype=dtype)),
            ak.index.Index(np.asarray([3, 4, 5, 9], dtype=dtype)),
            ak.contents.NumpyArray(np.arange(10)),
        )
    )
    array = ak.to_backend(source, "cuda")
    for index in (
        slice(None, None, -1),
        slice(1, None),
        slice(None, None, 2),
    ):
        assert array[:, index].to_list() == source[:, index].to_list()
    for index in (
        [[1, 0], [], [-1, 0], [1]],
        [[0, None, -1], [], [None, 1], [0, 1, None]],
    ):
        assert array[index].to_list() == source[index].to_list()
    assert ak.pad_none(array, 4).to_list() == ak.pad_none(source, 4).to_list()
    assert ak.is_valid(array)
    for index in (0, [0, 99]):
        with pytest.raises(IndexError):
            array[:, index]

    with pytest.raises(IndexError):
        array[[[0], [0], [0], [0]]]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("error_at", [None, 17])
def test_prepared_jagged_long_parallel_output(dtype, error_at):
    rows, width = 32, 4096
    starts = (3 + np.arange(rows) * width).astype(dtype)
    stops = (starts + width).astype(dtype)
    if error_at is not None:
        stops[error_at] -= 1
    output = np.full(rows * width + 3, 123, dtype=np.int64)
    args = [
        output.copy(),
        output.copy(),
        np.arange(width + 1, dtype=np.int64) * 2,
        output.copy(),
        starts,
        stops,
        width,
        rows,
    ]
    key = (
        "awkward_ListArray_getitem_jagged_expand",
        *(x.dtype.type for x in args if isinstance(x, np.ndarray)),
    )
    expected = NumpyBackend.instance()[key](*args)
    with cp.cuda.Stream(non_blocking=True):
        gpu = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
        # Reset outputs: CPU execution above may have populated their prefixes.
        for i in (0, 1, 3):
            gpu[i].fill(123)
        actual = CupyBackend.instance()[key](*gpu)
        contract.assert_error(actual, expected)
        for i in (0, 1, 3):
            np.testing.assert_array_equal(cp.asnumpy(gpu[i]), args[i])


@pytest.mark.parametrize(
    "offsets",
    [[0, 4096, 8192], [3, 3, 5, 6], [0, 8, 2, 10], [*range(33), 5, *range(34, 66)]],
)
def test_prepared_spread_parallel_and_overlap(offsets):
    output = np.full(max(offsets) + 3, 123, dtype=np.int64)
    args = [
        output,
        np.arange(len(offsets) - 1, dtype=np.int64) + 7,
        np.asarray(offsets, dtype=np.int64),
        len(offsets) - 1,
    ]
    key = (
        "awkward_ListArray_getitem_next_range_spreadadvanced",
        np.int64,
        np.int64,
        np.int64,
    )
    NumpyBackend.instance()[key](*args)
    with cp.cuda.Stream(non_blocking=True):
        gpu = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
        gpu[0].fill(123)
        CupyBackend.instance()[key](*gpu)
        np.testing.assert_array_equal(cp.asnumpy(gpu[0]), output)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("rows", [32, 129])
@pytest.mark.parametrize("bad_kind", [0, 1, 2, 3])
def test_prepared_advanced_first_error_and_rebinding(dtype, rows, bad_kind):
    starts = (np.arange(rows) * 8).astype(dtype)
    stops = starts + 8
    indices = np.asarray([0, -1, 8], dtype=np.int64)
    previous = (np.arange(rows) % 2).astype(np.int64)
    if bad_kind:
        row = min(rows - 1, 64)
        if bad_kind == 1:
            stops[row] = starts[row] - 1
        elif bad_kind == 2:
            stops[row] = rows * 8 + 1
        else:
            previous[row] = 2
        if row < rows - 1:
            stops[-1] = starts[-1] - 1
    output = np.full(rows + 3, 123, dtype=np.int64)
    args = [
        output.copy(),
        output.copy(),
        starts,
        stops,
        indices,
        previous,
        rows,
        rows * 8,
    ]
    key = (
        "awkward_ListArray_getitem_next_array_advanced",
        *(x.dtype.type for x in args if isinstance(x, np.ndarray)),
    )
    expected = NumpyBackend.instance()[key](*args)
    with cp.cuda.Stream(non_blocking=True):
        gpu = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
        gpu[0].fill(123)
        gpu[1].fill(123)
        actual = CupyBackend.instance()[key](*gpu)
        contract.assert_error(actual, expected)
        np.testing.assert_array_equal(cp.asnumpy(gpu[0]), args[0])
        np.testing.assert_array_equal(cp.asnumpy(gpu[1]), args[1])


@pytest.mark.parametrize("chunk_size", [1, 4, 16, 64])
def test_validation_tuning_candidates_gpu(chunk_size):
    from awkward._connect.cuda._listarray import _prepared_error, _Stream
    from awkward._connect.cuda._listarray_prepared import run

    rows = 129
    starts = np.arange(rows, dtype=np.int64) * 8
    stops = starts + 8
    indices = np.asarray([0, -1, 8], dtype=np.int64)
    previous = np.arange(rows, dtype=np.int64) % 2
    previous[64] = 2
    stops[-1] = starts[-1] - 1
    output = np.full(rows + 3, 123, dtype=np.int64)
    args = [
        output.copy(),
        output.copy(),
        starts,
        stops,
        indices,
        previous,
        rows,
        rows * 8,
    ]
    key = ("awkward_ListArray_getitem_next_array_advanced", *([np.int64] * 6))
    expected = NumpyBackend.instance()[key](*args)
    with cp.cuda.Stream(non_blocking=True):
        gpu = tuple(cp.asarray(x) for x in args[:6])
        gpu[0].fill(123)
        gpu[1].fill(123)
        status = run("advanced", gpu, rows, rows * 8, _Stream(), chunk_size=chunk_size)
        actual = _prepared_error(
            status,
            rows,
            ("stops[i] < starts[i]", "stops[i] > len(content)", "index out of range"),
            key[0],
        )
        contract.assert_error(actual, expected)
        for i in (0, 1):
            np.testing.assert_array_equal(cp.asnumpy(gpu[i]), args[i])


@pytest.mark.parametrize("offsets", [[0, 4096, 8192], [0, 8, 2, 10], list(range(66))])
def test_tiled_spread_tuning_candidate_gpu(offsets):
    from awkward._connect.cuda._listarray import _Stream
    from awkward._connect.cuda._listarray_prepared import run

    output = np.full(max(offsets) + 3, 123, dtype=np.int64)
    args = [
        output,
        np.arange(len(offsets) - 1, dtype=np.int64) + 7,
        np.asarray(offsets, dtype=np.int64),
        len(offsets) - 1,
    ]
    key = ("awkward_ListArray_getitem_next_range_spreadadvanced", *([np.int64] * 3))
    NumpyBackend.instance()[key](*args)
    with cp.cuda.Stream(non_blocking=True):
        gpu = tuple(cp.asarray(x) for x in args[:3])
        gpu[0].fill(123)
        run("spread", gpu, args[3], 0, _Stream(), spread_lanes=32)
        np.testing.assert_array_equal(cp.asnumpy(gpu[0]), output)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("rows", [1, 64, 65, 129])
@pytest.mark.parametrize(
    "failure", [None, "index", "slice_stop", "slice_reverse", "reverse", "content"]
)
def test_prepared_apply_chunk_boundaries_gpu(dtype, rows, failure):
    # Two launches with different contents exercise descriptor rebinding.
    with cp.cuda.Stream(non_blocking=True) as stream:
        for seed in range(2):
            error_at = min(64, rows - 1) if failure else None
            # Avoid unsigned underflow in source-bound failure fixtures.
            args = contract.slicing_apply_arguments(rows, 4, error_at, failure=failure)
            args[7] += 4 + seed
            args[8] += 4 + seed
            args[7] = args[7].astype(dtype)
            args[8] = args[8].astype(dtype)
            args[9] = rows * 4 + 16
            if failure == "content":
                args[8][error_at] = args[9] + 1
            key = (
                "awkward_ListArray_getitem_jagged_apply",
                *(x.dtype.type for x in args if isinstance(x, np.ndarray)),
            )
            gpu = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
            expected = NumpyBackend.instance()[key](*args)
            actual = CupyBackend.instance()[key](*gpu)
            contract.assert_error(actual, expected)
            for i in (0, 1):
                np.testing.assert_array_equal(cp.asnumpy(gpu[i]), args[i])
            ak._connect.cuda.synchronize_cuda(stream)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("rows,width", [(32, 0), (32, 8), (4096, 8), (32, 4096)])
@pytest.mark.parametrize(
    "start,stop,step",
    [
        (ak._util.kSliceNone, ak._util.kSliceNone, 1),
        (ak._util.kSliceNone, ak._util.kSliceNone, -1),
        (-7, 99, 3),
        (99, -99, -2),
    ],
)
def test_prepared_range_parallel_gpu(dtype, rows, width, start, stop, step):
    with cp.cuda.Stream(non_blocking=True) as stream:
        starts = (np.arange(rows) * width + 3).astype(dtype)
        stops = starts + width
        args = [
            np.full(rows + 4, 123, dtype=dtype),
            np.full(rows * width + 8, 123, dtype=np.int64),
            starts,
            stops,
            rows,
            start,
            stop,
            step,
        ]
        key = ("awkward_ListArray_getitem_next_range", dtype, np.int64, dtype, dtype)
        gpu = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
        expected = NumpyBackend.instance()[key](*args)
        actual = CupyBackend.instance()[key](*gpu)
        contract.assert_error(actual, expected)
        for i in (0, 1):
            np.testing.assert_array_equal(cp.asnumpy(gpu[i]), args[i])
        ak._connect.cuda.synchronize_cuda(stream)


@pytest.mark.parametrize("bad_index", [None, 0, 65536, 131071])
def test_few_rows_long_index_array(bad_index):
    rows, width = 4, 131072
    indices = np.arange(width, dtype=np.int64)
    if bad_index is not None:
        indices[bad_index] = width
    starts = np.arange(rows, dtype=np.int64) * width
    expected = [np.full(rows * width + 3, 123, dtype=np.int64) for _ in range(2)]
    args = [*expected, starts, starts + width, indices, rows, width, rows * width]
    device = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
    key = ("awkward_ListArray_getitem_next_array", *([np.int64] * 5))
    error = NumpyBackend.instance()[key](*args)
    actual = CupyBackend.instance()[key](*device)
    contract.assert_error(actual, error)
    for i in (0, 1):
        np.testing.assert_array_equal(cp.asnumpy(device[i]), expected[i])


def test_varying_shapes_reuse_prepared_setup():
    from awkward._connect.cuda import _listarray_prepared as prepared
    from awkward._connect.cuda import _listarray_slicing as slicing

    prepared._setup.cache_clear()
    slicing._setup.cache_clear()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for rows, width in [(2, 3), (129, 4), (4, 5), (129, 4)]:
            cpu = ak.Array(
                ak.contents.ListArray(
                    ak.index.Index64(np.arange(rows, dtype=np.int64) * width),
                    ak.index.Index64(np.arange(1, rows + 1, dtype=np.int64) * width),
                    ak.contents.NumpyArray(np.arange(rows * width)),
                )
            )
            array = ak.to_backend(cpu, "cuda")
            for slc in (slice(None, None, -1), slice(1, None), slice(None, None, 2)):
                assert array[:, slc].to_list() == cpu[:, slc].to_list()
            jagged = [[0, -1] for _ in range(rows)]
            assert array[jagged].to_list() == cpu[jagged].to_list()
        stream.synchronize()
    # Slice parameters and lengths must not be setup/JIT specialization keys.
    assert slicing._setup.cache_info().misses == 2  # apply and range
    assert slicing._setup.cache_info().hits > 2


def test_decreasing_padding_offsets_guard_buffers():
    from awkward._connect.cuda._listarray import rpad

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        for dtype in (np.int32, np.uint32, np.int64):
            guard = cp.full(8, 123, dtype=cp.int64)
            starts, stops = cp.asarray([5], dtype=dtype), cp.asarray([2], dtype=dtype)
            outstarts, outstops = cp.empty(1, dtype=dtype), cp.empty(1, dtype=dtype)
            rpad(guard[3:5], starts, stops, outstarts, outstops, 2, 1)
            np.testing.assert_array_equal(
                cp.asnumpy(guard), [123, 123, 123, -1, -1, 123, 123, 123]
            )


def test_returned_error_public_format():
    key = ("awkward_ListArray_getitem_next_at", *([np.int64] * 3))
    kernel = CupyBackend.instance()[key]
    error = kernel(
        cp.empty(1, dtype=cp.int64),
        cp.asarray([0], dtype=cp.int64),
        cp.asarray([2], dtype=cp.int64),
        1,
        5,
    )
    message = CupyBackend.instance().format_kernel_error(error)
    assert message.endswith(
        "(in compiled code: cuda.compute: awkward_ListArray_getitem_next_at)"
    )
    assert "while attempting to get index 5" in message


def test_prepared_jagged_apply_public_error_format():
    key = ("awkward_ListArray_getitem_jagged_apply", *([np.int64] * 7))
    kernel = CupyBackend.instance()[key]
    arrays = [
        cp.full(4, 123, dtype=cp.int64),
        cp.full(3, 123, dtype=cp.int64),
        cp.asarray([1, 2, 0], dtype=cp.int64),
        cp.asarray([0, 2, 5], dtype=cp.int64),
        cp.asarray([2, 5, 6], dtype=cp.int64),
        cp.asarray([0, 2, 2], dtype=cp.int64),
        cp.asarray([2, 2, 3], dtype=cp.int64),
    ]
    error = kernel(*arrays, 3, 3, 6)
    assert (error.id, error.attempt) == (0, 2)
    assert CupyBackend.instance().format_kernel_error(error) == (
        "index out of range while attempting to get index 2 "
        "(in compiled code: cuda.compute: awkward_ListArray_getitem_jagged_apply)"
    )
