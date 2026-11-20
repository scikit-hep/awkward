# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

from awkward._backends.numpy import NumpyBackend


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int64, np.uint64, np.float32, np.float64]
)
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize(
    "offsets,count,length",
    [
        ([], 0, 5),
        ([0], 1, 0),
        ([0] * 514, 514, 9),
        ([0, 0, 1, 1, 2, 3, 99], 6, 7),
        (list(range(514)), 514, 1024),
    ],
)
def test_trivial_segments_preserve_coverage(
    dtype, ascending, stable, offsets, count, length
):
    data = np.resize(
        np.asarray([True, False, True, False, False], dtype=dtype), max(1, length)
    )
    if np.dtype(dtype).kind == "f" and length > 1:
        data[0] = np.nan
        data[1] = -0.0
    covered = offsets[count - 1] if count else 0
    boundaries = np.asarray(offsets, dtype=np.int64)
    for op, offset_dtype in [
        ("sort", np.int64),
        ("argsort", np.int32),
        ("argsort", np.uint32),
        ("argsort", np.int64),
        ("argsort", np.uint64),
    ]:
        output_dtype = dtype if op == "sort" else np.int64
        output = np.full(length + 3, 99, dtype=output_dtype)
        expected = output.copy()
        if op == "sort":
            expected[:covered] = data[:covered]
        else:
            expected[:length] = np.arange(length)
            expected[:covered] = 0
        error = NumpyBackend.instance()[
            (f"awkward_{op}", output_dtype, dtype, offset_dtype)
        ](
            output,
            data,
            length,
            boundaries.astype(offset_dtype),
            count,
            ascending,
            stable,
        )
        assert error.str is None
        np.testing.assert_array_equal(output, expected)
        if op == "sort" and np.dtype(dtype).kind == "f" and covered > 1:
            assert np.signbit(output[1])


def test_nontrivial_segment_disables_trivial_path():
    data = np.asarray([9, 7, 4, 8, 1, 3], dtype=np.int64)
    for position in (1, 255, 256, 257, 513):
        offsets = np.zeros(position + 3, dtype=np.int64)
        offsets[position:] = [1, 3, 6]
        output = np.full(9, 99, dtype=np.int64)
        error = NumpyBackend.instance()[("awkward_sort", np.int64, np.int64, np.int64)](
            output, data, len(data), offsets, len(offsets), True, True
        )
        assert error.str is None
        np.testing.assert_array_equal(output, [9, 4, 7, 1, 3, 8, 99, 99, 99])
        error = NumpyBackend.instance()[
            ("awkward_argsort", np.int64, np.int64, np.int64)
        ](output, data, len(data), offsets, len(offsets), True, True)
        assert error.str is None
        np.testing.assert_array_equal(output, [0, 1, 0, 1, 2, 0, 99, 99, 99])
