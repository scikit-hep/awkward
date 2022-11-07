# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401
import cupy

numba = pytest.importorskip("numba")

from numba import cuda, types  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402

from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False

ak_numba = pytest.importorskip("awkward.numba")
ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")
ak_numba_layout = pytest.importorskip("awkward._connect.numba.layout")

ak.numba.register_and_check()

class ArrayViewArgHandler:
    def prepare_args(self, ty, val, **kwargs):
        print(repr(val), type(val))
        if isinstance(val, ak.Array):   
            return ty, val
        elif isinstance(val, ak._connect.numba.arrayview.ArrayView):
            return types.uint64, val._numbaview.lookup.arrayptrs
        else:
            return ty, val

array_view_arg_handler = ArrayViewArgHandler()

# FIXME: configure the blocks
# threadsperblock = 32
# blockspergrid = 128

@cuda.jit(extensions=[array_view_arg_handler])
def swallow(array):
    pass


@cuda.jit
def passthrough(array):
    return array


@cuda.jit
def passthrough2(array):
    return array, array


@cuda.jit
def digest(array):
    return array[0]


@cuda.jit
def digest2(array):
    tmp = array[0]
    return tmp, tmp, array[0]

def test_numpy_array_1d():
    nparray = np.array([0, 1, 2, 3], dtype=int)
    swallow[1, 1](nparray)

def test_to_numy_array_1d():
    akarray = ak.Array([0, 1, 2, 3])
    swallow[1, 1](ak.to_numpy(akarray))

#def test_array_1d():
#    akarray = ak.Array([0, 1, 2, 3])
#    swallow[1, 1](akarray))
    
def test_array_njit():
    @numba.njit
    def something(array):
        for index in range(len(array)):
            print(array[index])  # noqa: T201

    akarray = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    something(akarray)


def test_array_cuda_jit_cuda_backend():
    @numba.cuda.jit
    def something(array):
        index = numba.cuda.threadIdx.x
        if index > len(array):
            return

    akarray = ak.Array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], backend="cuda"
    )
    something[1, 1](ak.to_cupy(akarray))


def test_array_cuda_jit():
    @numba.cuda.jit
    def something(array):
        index = numba.cuda.threadIdx.x
        if index > len(array):
            return

    akarray = ak.Array(
        [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    )
    something[1, 1](ak.to_numpy(akarray))


