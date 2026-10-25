# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("out", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("op", ["sum", "prod"])
def test_cuda_defined_integer_wrapping(out, narrow, op):
    from awkward._backends.cupy import CupyBackend

    inp = (np.int8 if np.dtype(out).kind == "i" else np.uint8) if narrow else out
    info = np.iinfo(inp)
    rows = [
        [info.max, 1],
        [info.min, -1 if info.min < 0 else 0],
        [info.max, 2],
        [info.min, 2],
        [info.max] * 10,
        [],
    ]
    values = cp.asarray([99] + [x for row in rows for x in row] + [88], dtype=inp)
    offsets = cp.asarray(
        [1] + [1 + sum(map(len, rows[: i + 1])) for i in range(len(rows))],
        dtype=cp.int64,
    )
    result = cp.empty(len(rows), dtype=out)
    kernel = CupyBackend.instance()[f"awkward_reduce_{op}", out, inp, np.int64]
    kernel(result, values, offsets, len(rows))
    modulus = 1 << (np.dtype(out).itemsize * 8)
    expected = []
    for row in rows:
        value = (sum(row) if op == "sum" else math.prod(row)) % modulus
        if np.dtype(out).kind == "i" and value >= modulus // 2:
            value -= modulus
        expected.append(value)
    np.testing.assert_array_equal(cp.asnumpy(result), expected)
