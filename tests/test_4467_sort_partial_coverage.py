# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from awkward._backends.numpy import NumpyBackend

CASES = [
    ([], 0),
    ([0], 1),
    ([0, 0, 0], 3),
    ([0, 2], 2),
    ([0, 2, 2, 4], 4),
    ([0, 2, 5], 2),
]


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).resolve().parents[1] / "awkward-cpp/tests-spec/kernels.py"
    if not path.exists():
        pytest.skip("requires generated kernel references")
    spec = importlib.util.spec_from_file_location("coverage_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("op", ["sort", "argsort"])
@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int64, np.uint64, np.float32, np.float64]
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("offsets,count", CASES)
def test_cpu_and_reference_coverage(reference, op, dtype, ascending, offsets, count):
    values = np.asarray([3, 1, 2, 0, 99], dtype=dtype)
    boundaries = np.asarray(offsets, dtype=np.int64)
    output_type = dtype if op == "sort" else np.int64
    output = np.full(7, 1, dtype=output_type)
    expected = output.tolist()
    oracle = getattr(reference, f"awkward_{op}")
    oracle(expected, values.tolist(), 5, offsets, count, ascending, True)
    key = (
        ("awkward_sort", dtype, dtype, np.int64)
        if op == "sort"
        else ("awkward_argsort", np.int64, dtype, np.int64)
    )
    error = NumpyBackend.instance()[key](
        output, values, 5, boundaries, count, ascending, True
    )
    assert error.str is None
    np.testing.assert_array_equal(output, np.asarray(expected, dtype=output_type))
    np.testing.assert_array_equal(output[5:], [1, 1])


@pytest.mark.parametrize("op", ["sort", "argsort"])
def test_zero_length_buffers(op):
    values = np.empty(0, dtype=np.int64)
    offsets = np.asarray([0, 0], dtype=np.int64)
    output = np.full(2, -99, dtype=np.int64)
    key = (f"awkward_{op}", np.int64, np.int64, np.int64)
    assert (
        NumpyBackend.instance()[key](output, values, 0, offsets, 2, True, True).str
        is None
    )
    np.testing.assert_array_equal(output, [-99, -99])
