# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import re

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("op", ["sort", "argsort"])
@pytest.mark.parametrize(
    "offsets,count,length,message",
    [
        ([], 0, -1, "length must be non-negative"),
        ([0], -1, 5, "offsetslength must be non-negative"),
        ([1, 2], 2, 5, "offsets[0] must be zero"),
        ([0, 3, 2], 3, 5, "offsets must be nondecreasing"),
        ([0, 6], 2, 5, "offsets exceed length"),
        ([0, 6, -1], 3, 5, "offsets exceed length"),
        ([-1, 2], 2, 5, "offsets must be non-negative"),
        ([0, -1, 9], 3, 5, "offsets must be non-negative"),
        ([0, -(2**63)], 2, 5, "offsets must be non-negative"),
    ],
)
def test_invalid_offsets_fail_before_output_writes(op, offsets, count, length, message):
    from awkward._backends.cupy import CupyBackend

    values = cp.asarray([3, 1, 2, 0, 9], dtype=cp.int64)
    boundaries = cp.asarray(offsets, dtype=cp.int64)
    output = cp.full(7, -99, dtype=cp.int64)
    key = (f"awkward_{op}", np.int64, np.int64, np.int64)
    with pytest.raises(ValueError, match=re.escape(message)):
        CupyBackend.instance()[key](
            output, values, length, boundaries, count, True, False
        )
    np.testing.assert_array_equal(cp.asnumpy(output), np.full(7, -99))


@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_unsigned_offsets_rejected_before_narrowing(value):
    from awkward._backends.cupy import CupyBackend

    output = cp.full(7, -99, dtype=cp.int64)
    offsets = cp.asarray([0, value], dtype=cp.uint64)
    values = cp.asarray([3, 1, 2, 0, 9], dtype=cp.int64)
    key = ("awkward_argsort", np.int64, np.int64, np.uint64)
    with pytest.raises(ValueError, match="offsets exceed length"):
        CupyBackend.instance()[key](output, values, 5, offsets, 2, True, False)
    np.testing.assert_array_equal(cp.asnumpy(output), np.full(7, -99))
