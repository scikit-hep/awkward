# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

from numba.extending import overload_method
from numba import cuda, types

from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False

ak_numba = pytest.importorskip("awkward.numba")
ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")
ak_numba_layout = pytest.importorskip("awkward._connect.numba.layout")

ak.numba.register_and_check()

@overload_method(types.Array, 'sum', target='cuda')
def array_sum(arr):
    if arr.ndim != 1:
        # Only implement 1D for this quick example
        return None

    def sum_impl(arr):
        res = 0 
        for i in range(len(arr)):
            res += arr[i]
        return res 
    return sum_impl

@cuda.jit
def f(arr):
    print("Sum is", arr.sum())


def test_overload_method():
    f[1, 1](np.arange(5))
    f[1, 1](np.arange(10))
    cuda.synchronize()
