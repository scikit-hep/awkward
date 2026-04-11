from __future__ import annotations

import cupy
import cupy.testing as cpt
import pytest

import awkward._connect.cuda as ak_cu
from awkward._backends.cupy import CupyBackend

cupy_backend = CupyBackend.instance()


# tests for `missing_repeat` that are mising at `tests-cuda-kernels-explicit`
# (with multiple repetitions and regularsize > 1)
def test_unit_cudaawkward_missing_repeat_64_repetitions_2():
    outindex = cupy.array(
        [123, 123, 123, 123, 123, 123, 123, 123, 123, 123], dtype=cupy.int64
    )
    index = cupy.array([1, -1, 1, 1, 0], dtype=cupy.int64)
    indexlength = 5
    regularsize = 1
    repetitions = 2
    funcC = cupy_backend["awkward_missing_repeat", cupy.int64, cupy.int64]
    funcC(outindex, index, indexlength, repetitions, regularsize)

    try:
        ak_cu.synchronize_cuda()
    except Exception as e:
        if "not implemented for given n" in str(e):
            print(
                "Not implemented for given n in compiled CUDA code (awkward_ListArray_combinations)"
            )
        else:
            pytest.fail(
                f"Unexpected error raised: {e}: This test case shouldn't have raised an error"
            )
    pytest_outindex = [1, -1, 1, 1, 0, 2, -1, 2, 2, 1]
    cpt.assert_allclose(outindex[: len(pytest_outindex)], cupy.array(pytest_outindex))


def test_unit_cudaawkward_missing_repeat_64_repetitions_3():
    outindex = cupy.array([123] * 15, dtype=cupy.int64)
    index = cupy.array([1, -1, 1, 1, 0], dtype=cupy.int64)
    indexlength = 5
    regularsize = 2
    repetitions = 3
    funcC = cupy_backend["awkward_missing_repeat", cupy.int64, cupy.int64]
    funcC(outindex, index, indexlength, repetitions, regularsize)

    try:
        ak_cu.synchronize_cuda()
    except Exception as e:
        if "not implemented for given n" in str(e):
            print(
                "Not implemented for given n in compiled CUDA code (awkward_ListArray_combinations)"
            )
        else:
            pytest.fail(
                f"Unexpected error raised: {e}: This test case shouldn't have raised an error"
            )
    pytest_outindex = [1, -1, 1, 1, 0, 3, -1, 3, 3, 2, 5, -1, 5, 5, 4]
    cpt.assert_allclose(outindex[: len(pytest_outindex)], cupy.array(pytest_outindex))
