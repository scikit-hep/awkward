# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

from numba import cuda  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402

ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")

ak.numba.register_and_check()


def test_array_njit():
    @numba.njit
    def something(array):
        for index in range(len(array)):
            print(array[index])  # noqa: T201

    akarray = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    something(akarray)


def test_array_cuda_jit_cuda_backend():
    @numba.cuda.jit
    def something(array):
        index = numba.cuda.threadIdx.x
        if index > len(array):
            return

    akarray = ak.Array(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]], backend="cuda"
    )
    threadsperblock = 32
    blockspergrid = 128
    something[blockspergrid, threadsperblock](akarray)


def test_array_cuda_jit():
    @numba.cuda.jit
    def something(array):
        index = numba.cuda.threadIdx.x
        if index > len(array):
            return

    akarray = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    threadsperblock = 32
    blockspergrid = 128
    something[blockspergrid, threadsperblock](akarray)
