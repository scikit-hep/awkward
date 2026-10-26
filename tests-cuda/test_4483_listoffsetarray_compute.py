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
    "listoffsetarray_contract_cases",
    Path(__file__).resolve().parents[1] / "tests/test_4483_listoffsetarray_compute.py",
)
contract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = contract
spec.loader.exec_module(contract)


@pytest.mark.parametrize("name,specialization,case", contract.CASES)
def test_cuda_cpu_parity(name, specialization, case):
    with cp.cuda.Stream(non_blocking=True):
        _cuda_cpu_parity(name, specialization, case)


def _cuda_cpu_parity(name, specialization, case):
    args, outputs, key = contract.arguments(specialization, case)
    gpu_args = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
    expected = NumpyBackend.instance()[(name, *key)](*args)
    kernel = CupyBackend.instance()[(name, *key)]
    assert isinstance(kernel, ak._kernels.CudaComputeKernel)
    actual = kernel(*gpu_args)
    contract.assert_error(actual, expected)
    for i in outputs:
        np.testing.assert_array_equal(cp.asnumpy(gpu_args[i]), args[i])


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_highlevel_operations(dtype):
    with cp.cuda.Stream(non_blocking=True):
        offsets = ak.index.Index(np.asarray([2, 4, 4, 7], dtype=dtype))
        source = ak.Array(
            ak.contents.ListOffsetArray(offsets, ak.contents.NumpyArray(np.arange(9)))
        )
        array = ak.to_backend(source, "cuda")
        assert ak.pad_none(array, 4).to_list() == ak.pad_none(source, 4).to_list()
        assert ak.flatten(array).to_list() == ak.flatten(source).to_list()
        assert (
            ak.drop_none(ak.mask(array, array % 2 == 0), axis=-1).to_list()
            == ak.drop_none(ak.mask(source, source % 2 == 0), axis=-1).to_list()
        )
        regular = ak.to_backend([[1, 2], [3, 4]], "cuda")
        assert ak.to_regular(regular).to_list() == [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="not regular"):
            ak.to_regular(array)


@pytest.mark.parametrize("groups,width", [(1, 1), (17, 33), (257, 513)])
def test_nonlocal_ranges_across_block_boundaries(groups, width):
    rng = np.random.default_rng(2026)
    distincts = rng.choice([-1, -2, 0, 7], groups * width).astype(np.int64)
    starts = np.full(groups + 3, 123, dtype=np.int64)
    stops = starts.copy()
    args = [starts, stops, distincts, distincts.size, groups]
    key = (
        "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
        np.int64,
        np.int64,
        np.int64,
    )
    with cp.cuda.Stream(non_blocking=True):
        gpu_args = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
        NumpyBackend.instance()[key](*args)
        CupyBackend.instance()[key](*gpu_args)
        np.testing.assert_array_equal(cp.asnumpy(gpu_args[0]), starts)
        np.testing.assert_array_equal(cp.asnumpy(gpu_args[1]), stops)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("offsets", [[0], [2, 4, 4, 7]])
@pytest.mark.parametrize("target", [0, 2, 4])
def test_padding_offset_widths(dtype, offsets, target):
    source = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index(np.asarray(offsets, dtype=dtype)),
            ak.contents.NumpyArray(np.arange(9)),
        )
    )
    expected = ak.pad_none(source, target)
    with cp.cuda.Stream(non_blocking=True):
        result = ak.pad_none(ak.to_backend(source, "cuda"), target)
        assert result.to_list() == expected.to_list()
        assert result.layout.offsets.dtype == np.dtype(np.int64)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_prepared_gather_rebinds_buffers_and_streams(dtype):
    from awkward._connect.cuda._listoffsetarray import flatten_offsets

    pending = []
    # Submit all launches before synchronizing, with different shapes, origins,
    # pointers and streams. Prepared state must be copied into each launch.
    for length in (3, 7, 2, 9):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            outer = cp.arange(1, length + 1, dtype=dtype)
            inner = cp.arange(length + 2, dtype=cp.int64) + length * 10
            storage = cp.full(length + 4, -1, dtype=cp.int64)
            flatten_offsets(storage[1:], outer, length, inner)
        pending.append((stream, outer, inner, storage, length))
    for stream, _outer, _inner, storage, length in pending:
        stream.synchronize()
        expected = np.full(length + 4, -1, dtype=np.int64)
        expected[1 : length + 1] = np.arange(1, length + 1) + length * 10
        np.testing.assert_array_equal(cp.asnumpy(storage), expected)


@pytest.mark.parametrize("width", [1, 8, 4096])
def test_prepared_nonlocal_ranges_rebinds_buffers_and_streams(width):
    from awkward._connect.cuda._listoffsetarray import nonlocal_ranges

    pending = []
    for groups in (3, 7, 4):
        source = np.full(groups * width, -1, dtype=np.int64)
        source[width] = -2
        source[3 * width - 1] = 0
        expected_starts = np.full(groups + 3, 123, dtype=np.int64)
        expected_stops = expected_starts.copy()
        NumpyBackend.instance()[
            "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
            np.int64,
            np.int64,
            np.int64,
        ](expected_starts, expected_stops, source, source.size, groups)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            distincts = cp.asarray(source)
            starts = cp.full(groups + 3, 123, dtype=cp.int64)
            stops = starts.copy()
            nonlocal_ranges(starts, stops, distincts, source.size, groups)
        pending.append(
            (stream, distincts, starts, stops, expected_starts, expected_stops)
        )
    for stream, _distincts, starts, stops, expected_starts, expected_stops in pending:
        stream.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(starts), expected_starts)
        np.testing.assert_array_equal(cp.asnumpy(stops), expected_stops)
