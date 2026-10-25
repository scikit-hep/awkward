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
