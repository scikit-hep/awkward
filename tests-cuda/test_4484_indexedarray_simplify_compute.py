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
    "indexedarray_simplify_cases",
    Path(__file__).resolve().parents[1]
    / "tests/test_4484_indexedarray_simplify_compute.py",
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


@pytest.mark.parametrize("outer_dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("inner_dtype", [np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("option", [False, True])
def test_public_simplification(outer_dtype, inner_dtype, option):
    inner_class = (
        ak.contents.IndexedOptionArray
        if option and inner_dtype != np.uint32
        else ak.contents.IndexedArray
    )
    inner_values = (
        [2, -1, 0] if inner_class is ak.contents.IndexedOptionArray else [2, 1, 0]
    )
    inner = inner_class(
        ak.index.Index(np.asarray(inner_values, dtype=inner_dtype)),
        ak.contents.NumpyArray(np.asarray([10, 20, 30])),
    )
    outer_values = [1, -1, 0] if option and outer_dtype != np.uint32 else [1, 0, 2]
    outer_class = ak.contents.IndexedOptionArray if option else ak.contents.IndexedArray
    expected = outer_class.simplified(
        ak.index.Index(np.asarray(outer_values, dtype=outer_dtype)), inner
    ).to_list()
    with cp.cuda.Stream(non_blocking=True) as stream:
        gpu_inner = ak.to_backend(inner, "cuda", highlevel=False)
        outer = ak.index.Index(cp.asarray(outer_values, dtype=outer_dtype))
        assert outer_class.simplified(outer, gpu_inner).to_list() == expected
        with pytest.raises(ValueError, match="index out of range"):
            outer_class.simplified(
                ak.index.Index(cp.asarray([1, 3], dtype=outer_dtype)), gpu_inner
            )
        ak._connect.cuda.synchronize_cuda(stream)
        assert outer_class.simplified(outer, gpu_inner).to_list() == expected


@pytest.mark.parametrize(
    "length,error_at", [(33, None), (257, 0), (4097, 256), (4097, 4096)]
)
def test_first_error_across_blocks(length, error_at):
    args = contract.large_case(length, error_at)
    key = ("awkward_IndexedArray_simplify", np.int64, np.int64, np.int64)
    with cp.cuda.Stream(non_blocking=True):
        gpu_args = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in args]
        expected = NumpyBackend.instance()[key](*args)
        actual = CupyBackend.instance()[key](*gpu_args)
        contract.assert_error(actual, expected)
        np.testing.assert_array_equal(cp.asnumpy(gpu_args[0]), args[0])
