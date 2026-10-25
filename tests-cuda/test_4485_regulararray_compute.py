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
    "regulararray_contract_cases",
    Path(__file__).resolve().parents[1] / "tests/test_4485_regulararray_compute.py",
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


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("size", [0, 2, 5])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_combinations_cpu_parity(n, replacement, size, length):
    expected, expected_backing, _ = contract.combination_arguments(
        n, replacement, size, length
    )
    key = ("awkward_RegularArray_combinations_64", np.int64, np.int64, np.int64)
    assert NumpyBackend.instance()[key](*expected).str is None
    with cp.cuda.Stream(non_blocking=True):
        _, initial_backing, _ = contract.combination_arguments(
            n, replacement, size, length
        )
        backing = [cp.asarray(x) for x in initial_backing]
        carries = [x[2:] for x in backing]
        pointers = cp.asarray([x.data.ptr for x in carries], dtype=cp.intp)
        toindex = cp.full(n + 3, -99, dtype=cp.int64)
        fromindex = cp.arange(n + 3, dtype=cp.int64) + 17
        kernel = CupyBackend.instance()[key]
        assert isinstance(kernel, ak._kernels.CudaComputeKernel)
        kernel(pointers, toindex, fromindex, n, replacement, size, length)
        for actual, wanted in zip(backing, expected_backing, strict=True):
            np.testing.assert_array_equal(cp.asnumpy(actual), wanted)
        np.testing.assert_array_equal(cp.asnumpy(toindex), expected[1])
        np.testing.assert_array_equal(cp.asnumpy(fromindex), expected[2])


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("replacement", [False, True])
def test_public_regular_operations(n, replacement):
    source = ak.Array(
        ak.contents.RegularArray(
            ak.contents.NumpyArray(np.arange(12)),
            size=4,
        )
    )
    with cp.cuda.Stream(non_blocking=True):
        array = ak.to_backend(source, "cuda")
        # Multidimensional NumpyArray indexing delegates directly to CuPy and
        # bypasses the RegularArray regularization kernel under test.
        assert isinstance(array.layout, ak.contents.RegularArray)
        assert array.layout.content.data.ndim == 1
        assert (
            ak.combinations(array, n, replacement=replacement).to_list()
            == ak.combinations(source, n, replacement=replacement).to_list()
        )
        assert (
            ak.argcombinations(array, n, replacement=replacement).to_list()
            == ak.argcombinations(source, n, replacement=replacement).to_list()
        )
        assert array[:, [-1, 0, 1]].to_list() == source[:, [-1, 0, 1]].to_list()
        with pytest.raises(IndexError, match="index out of range"):
            array[:, [0, 4]]
