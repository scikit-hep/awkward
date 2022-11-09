# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

from numba import cuda, types  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402
from numba.extending import overload
from numba.extending import overload_method

from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False

ak_numba = pytest.importorskip("awkward.numba")
ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")
ak_numba_layout = pytest.importorskip("awkward._connect.numba.layout")

ak.numba.register_and_check()

class ArrayViewArgHandler:
    def prepare_args(self, ty, val, **kwargs):
        print("ArrayViewArgHandler::prepare_args", repr(val), type(val))
        if isinstance(val, ak.Array):
            print("High level ak.Array to_numpy data pointer is", ak.to_numpy(val).ctypes.data)
            # A pointer to the memory area of the array as a Python integer.
            # return types.uint64, val.layout._data.ctypes.data
            #
            return types.uint64, ak._connect.numba.arrayview.ArrayView.fromarray(val).lookup.arrayptrs.ctypes.data
        elif isinstance(val, ak._connect.numba.arrayview.ArrayView):
            print("ak.ArrayView")
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

@cuda.jit(extensions=[array_view_arg_handler])
def multiply(array, n):
    for i in range(len(array)):
        print(n * array[i])

@cuda.jit
def passthrough(array, device=True):
    return array


@cuda.jit
def passthrough2(array):
    return array, array


@cuda.jit
def digest(array, val):
    val[0] = 10 * array[0]


@cuda.jit
def digest2(array, tmp):
    tmp[0] = array[0]
    tmp[1] = array[0]
    tmp[2] = array[0]


@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1
        
def test_numpy_array_1d():
    nparray = np.array([10, 1, 2, 3], dtype=int)
    swallow[1, 1](nparray)
    print("Swallow", nparray)
    multiply[1, 1](nparray, 3)
    print("Multiply", nparray)
#    arr = passthrough[1, 1](np.array([10, 11, 12, 13], dtype=int))
#    arr1, arr2 = passthrough2[1, 1](arr)
#    print(arr1, arr2)
    v0 = np.empty(1)
    digest[1, 1](nparray, v0)
    print("Digest", v0[0])
    v1 = np.empty(3)
    digest2[1, 1](nparray, v1)
    print(v1)
    increment_by_one[1, 3](v1)
    print(v1)

def test_to_numy_array_1d():
    akarray = ak.Array([0, 1, 2, 3])
    swallow[1, 1](ak.to_numpy(akarray))

def test_mem_management():
    # copy host->device a numpy array:
    ary = np.arange(10)
    d_ary = cuda.to_device(ary)

    
def test_mem_management1():
    ary = np.arange(10)
    # enqueue the transfer to a stream:
    stream = cuda.stream()
    d_ary = cuda.to_device(ary, stream=stream)
    # copy device->host:
    hary = d_ary.copy_to_host()

def test_mem_management2():
    ary = np.arange(10)
    d_ary = cuda.to_device(ary)
    # copy device->host to an existing array:
    ary = np.empty(shape=d_ary.shape, dtype=d_ary.dtype)
    d_ary.copy_to_host(ary)
    
def test_array_1d():    
    akarray = ak.Array([0, 1, 2, 3])
    swallow[1, 1](akarray)

    
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


@overload(len, target='cuda')
def grid_group_len(seq):
    if isinstance(seq, cuda.types.GridGroup):
        def len_impl(seq):
            n = cuda.gridsize(1)
            return n
        return len_impl


@cuda.jit
def fun_len():
    if cuda.grid(1) == 0:
        print("Grid size is", len(cuda.cg.this_grid()))

def test_fun_overload():
    fun_len[1, 1]()
    fun_len[1, 2]()
    fun_len[1, 3]()
    cuda.synchronize()


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
def fun_sum(arr):
    print("Sum is", arr.sum())

import numpy as np

def test_method_overload():
    fun_sum[1, 1](np.arange(5))
    fun_sum[1, 1](np.arange(10))
    cuda.synchronize()
    
