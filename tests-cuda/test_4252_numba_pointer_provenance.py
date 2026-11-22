# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")
cuda = pytest.importorskip("numba.cuda")
ak.numba.register_and_check()
cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = False


@pytest.mark.parametrize("dtype", [np.bool_, np.int8, np.int64, np.float64])
def test_separate_pointer_table(dtype):
    @cuda.jit(extensions=[ak.numba.cuda])
    def copy(array, output):
        i = cuda.grid(1)
        if i < len(array):
            output[i] = array[i].x[-1]

    source = np.arange(6).astype(dtype)
    array = ak.Array(
        ak.contents.RecordArray(
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index64(np.array([0, 2, 3, 6])),
                    ak.contents.NumpyArray(source),
                )
            ],
            ["x"],
        )
    )
    array = ak.to_backend(array, "cuda")
    result = cp.empty(3, dtype=dtype)
    copy[1, 32](array, result)
    cp.cuda.Device().synchronize()
    cp.testing.assert_array_equal(result, cp.asarray(source[[1, 2, 5]]))
