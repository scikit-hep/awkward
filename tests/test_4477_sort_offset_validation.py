# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from awkward._backends.numpy import NumpyBackend

CASES = [
    ([], 0, -1, "length must be non-negative", 2**63 - 1),
    ([0], -1, 5, "offsetslength must be non-negative", 2**63 - 1),
    ([1, 2], 2, 5, "offsets[0] must be zero", 0),
    ([0, 3, 2], 3, 5, "offsets must be nondecreasing", 2),
    ([0, 6], 2, 5, "offsets exceed length", 1),
    ([0, 6, -1], 3, 5, "offsets exceed length", 1),
    ([-1, 2], 2, 5, "offsets must be non-negative", 0),
    ([0, -1, 9], 3, 5, "offsets must be non-negative", 1),
    ([0, -(2**63)], 2, 5, "offsets must be non-negative", 1),
    ([0, 2**63], 2, 5, "offsets exceed length", 1),
    ([0, 2**64 - 1], 2, 5, "offsets exceed length", 1),
]
PARAMS = [
    (op, dtype, case)
    for op, dtype in [
        ("sort", np.int64),
        ("argsort", np.int32),
        ("argsort", np.uint32),
        ("argsort", np.int64),
        ("argsort", np.uint64),
    ]
    for case in CASES
    if all(np.iinfo(dtype).min <= x <= np.iinfo(dtype).max for x in case[0])
]


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).resolve().parents[1] / "awkward-cpp/tests-spec/kernels.py"
    if not path.exists():
        pytest.skip("requires generated kernel references")
    spec = importlib.util.spec_from_file_location("offset_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("op,offset_dtype,case", PARAMS)
def test_cpu_invalid_offsets_do_not_write(op, offset_dtype, case):
    offsets, count, length, message, position = case
    values = np.asarray([3, 1, 2, 0, 9], dtype=np.int64)
    boundaries = np.asarray(offsets, dtype=offset_dtype)
    output = np.full(7, -99, dtype=np.int64)
    key = (f"awkward_{op}", np.int64, np.int64, offset_dtype)
    error = NumpyBackend.instance()[key](
        output, values, length, boundaries, count, True, False
    )
    assert error.str == message.encode()
    assert error.id == position
    assert error.attempt == 2**63 - 1
    assert f"awkward_{op}.cpp".encode() in error.filename
    np.testing.assert_array_equal(output, np.full(7, -99))


@pytest.mark.parametrize("op,offset_dtype,case", PARAMS)
def test_reference_invalid_offsets_do_not_write(reference, op, offset_dtype, case):
    offsets, count, length, message, _ = case
    output = [-99] * 7
    with pytest.raises(ValueError) as err:
        getattr(reference, f"awkward_{op}")(
            output, [3, 1, 2, 0, 9], length, offsets, count, True, False
        )
    assert str(err.value) == message
    assert output == [-99] * 7
