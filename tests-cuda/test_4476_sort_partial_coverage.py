# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("op", ["sort", "argsort"])
@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int64, np.uint64, np.float32, np.float64]
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize(
    "offsets,count",
    [([], 0), ([0], 1), ([0, 0, 0], 3), ([0, 2], 2), ([0, 2, 2, 4], 4), ([0, 2, 5], 2)],
)
def test_partial_sort_coverage(op, dtype, ascending, stable, offsets, count):
    from awkward._backends.cupy import CupyBackend

    values = np.asarray([3, 1, 2, 0, 99], dtype=dtype)
    output_type = dtype if op == "sort" else np.int64
    result = cp.full(7, 1, dtype=output_type)
    boundaries = cp.asarray(offsets, dtype=cp.int64)
    key = (
        ("awkward_sort", dtype, dtype, np.int64)
        if op == "sort"
        else ("awkward_argsort", np.int64, dtype, np.int64)
    )
    CupyBackend.instance()[key](
        result, cp.asarray(values), 5, boundaries, count, ascending, stable
    )
    actual = cp.asnumpy(result)
    covered = offsets[count - 1] if count else 0
    if op == "argsort":
        np.testing.assert_array_equal(actual[covered:5], np.arange(covered, 5))
    else:
        np.testing.assert_array_equal(actual[covered:5], np.ones(5 - covered))
    np.testing.assert_array_equal(actual[5:], [1, 1])
    for start, stop in zip(
        offsets[: count - 1] if count else [], offsets[1:count], strict=True
    ):
        expected = np.sort(values[start:stop])
        if not ascending:
            expected = expected[::-1]
        if op == "sort":
            np.testing.assert_array_equal(actual[start:stop], expected)
        else:
            indices = actual[start:stop]
            np.testing.assert_array_equal(np.sort(indices), np.arange(stop - start))
            np.testing.assert_array_equal(values[start:stop][indices], expected)
            if stable:
                order = sorted(
                    range(stop - start),
                    key=lambda i: values[start + i],
                    reverse=not ascending,
                )
                np.testing.assert_array_equal(indices, order)


@pytest.mark.parametrize("op", ["sort", "argsort"])
def test_zero_length_buffers(op):
    from awkward._backends.cupy import CupyBackend

    values = cp.empty(0, dtype=cp.int64)
    offsets = cp.asarray([0, 0], dtype=cp.int64)
    output = cp.full(2, -99, dtype=cp.int64)
    key = (f"awkward_{op}", np.int64, np.int64, np.int64)
    CupyBackend.instance()[key](output, values, 0, offsets, 2, True, True)
    np.testing.assert_array_equal(cp.asnumpy(output), [-99, -99])


@pytest.mark.parametrize("offset_dtype", [np.int32, np.uint32, np.uint64])
def test_argsort_declared_offset_prefix(offset_dtype):
    from awkward._backends.cupy import CupyBackend

    values = cp.asarray([3, 1, 8, 9, 10], dtype=cp.int64)
    offsets = cp.asarray([0, 2, 5], dtype=offset_dtype)
    output = cp.full(5, -99, dtype=cp.int64)
    key = ("awkward_argsort", np.int64, np.int64, offset_dtype)
    CupyBackend.instance()[key](output, values, 5, offsets, 2, True, True)
    np.testing.assert_array_equal(cp.asnumpy(output), [1, 0, 2, 3, 4])
