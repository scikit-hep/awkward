# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

from numba import cuda, types  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402
from numba.extending import overload
from numba.extending import overload_method
from numba.cuda.args import wrap_arg

from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = False
config.CUDA_WARN_ON_IMPLICIT_COPY = False

ak_numba = pytest.importorskip("awkward.numba")
ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")
ak_numba_layout = pytest.importorskip("awkward._connect.numba.layout")

ak.numba.register_and_check()

# FIXME: do the same for ak.Array -> ArrayView
def lower_const_view(context, builder, viewtype, view):
    pos = view.pos
    start = view.start
    stop = view.stop
    lookup = view.lookup
    arrayptrs = lookup.arrayptrs

    arrayptrs_val = context.make_constant_array(
        builder, numba.typeof(arrayptrs), arrayptrs
    )

    proxyout = context.make_helper(builder, viewtype)
    proxyout.pos = context.get_constant(numba.intp, pos)
    proxyout.start = context.get_constant(numba.intp, start)
    proxyout.stop = context.get_constant(numba.intp, stop)
    proxyout.arrayptrs = context.make_helper(
        builder, numba.typeof(arrayptrs), arrayptrs_val
    ).data
    proxyout.pylookup = context.add_dynamic_addr(
        builder, id(lookup), info=str(type(lookup))
    )

    return proxyout._getvalue()

def lower_const_array_as_const_view(val):
    print("-------- start lower_const_array_as_const_view -------")
    #array_view = ak._connect.numba.arrayview.ArrayView.fromarray(array)
    # val._numbaview.lookup, val._numbaview.pos, val._numbaview.start, val._numbaview.stop
    start = id(val._numbaview.start)
    stop = id(val._numbaview.stop)
    pos = id(val._numbaview.pos)
    arrayptrs = id(val._numbaview.lookup.arrayptrs)
    print("arrayptrs", arrayptrs, hex(arrayptrs), format(arrayptrs, 'x'))
    pylookup = id(val._numbaview.lookup)
    
    return (start, stop, pos, arrayptrs, pylookup)


def lower_as_const_view(array_view):

    start = id(array_view.start)
    stop = id(array_view.stop)
    pos = id(array_view.pos)
    arrayptrs = id(array_view.lookup.arrayptrs)
    pylookup = id(array_view.lookup)

    return (start, stop, pos, arrayptrs, pylookup)

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

# access statistics of these memory pools.
def print_mempool_stats(idx):
    print("-------------------------")
    print(idx, ": mempool.used_bytes", mempool.used_bytes())
    print(idx, ": mempool.total_bytes", mempool.total_bytes())
    print(idx, ": pinned_mempool.n_free_blocks", pinned_mempool.n_free_blocks())

class ArrayViewArgHandler:
    def prepare_args(self, ty, val, stream, retr):
        print(">>>>>>>> ArrayViewArgHandler::prepare_args", repr(val), type(val))

        if isinstance(val, int):
            result = format(val, 'x')
            print(result)
        if isinstance(val, ak.Array):
            print(ty, dir(ty), ty.type)
            print(val, dir(val), val._numbaview, dir(val._numbaview))
            print("val is ak.Array")

            if isinstance(val.layout.nplike, ak.nplikes.Cupy):
                print("CUDA-DA-DA-DA")

                # Use uint64 for start, stop, pos, the array pointers value and the pylookup value
                tys = types.UniTuple(types.uint64, 5)
                print("Already has an ArrayView:", val._numbaview.lookup, val._numbaview.pos, val._numbaview.start, val._numbaview.stop)
                # ... Copy data to device if necessary...
                # use similar to: wrap_arg(val).to_device(retr, stream) where
                #
                # def wrap_arg(value, default=InOut):
                #    return value if isinstance(value, ArgHint) else default(value)

                dev = cuda.current_context().device
                print(dev)

                # access statistics of these memory pools.
                print_mempool_stats(1)

                # dev_array = ak.to_backend(val, "cuda") # it does copy to device!
                # print(">>> AFTER COPY TO DEVICE! >>>", dev_array, repr(dev_array), type(dev_array))
                # print_mempool_stats(2)

                #            ary = np.arange(10)
                #            d_ary = cuda.to_device(ary)
                #            print(d_ary)

                #start, stop, pos, arrayptrs, pylookup = lower_const_array_as_const_view(val)
                # val._numbaview.lookup, val._numbaview.pos, val._numbaview.start, val._numbaview.stop
 
                start = val._numbaview.start
                stop = val._numbaview.stop
                pos = val._numbaview.pos
                arrayptrs = val._numbaview.lookup.arrayptrs.data.ptr
                pylookup = id(val._numbaview.lookup)
                
                # Retrieve data back to host
                #            def retrieve():
                # Copy dev_array back to host somehow
                #                hary = ak.to_backend(dev_array, "cpu") # d_ary.copy_to_host()
                #                print("retrieve", hary, repr(hary), type(hary))
                #                print_mempool_stats(3)

                #                return hary # ak.to_backend(dev_array, "cpu") 
            
                # Append retrieve function if necessary
                #            retr.append(retrieve)
                print("about to return from prepare args")
                return tys, (start, stop, pos, arrayptrs, pylookup)
            else:
                raise ak._errors.wrap_error(NotImplementedError (
                    f"{repr(val.layout.nplike)} is not implemented for CUDA. Please transfer the array to CUDA backend to "
                    "continue the operation."))
            
        else:
            return ty, val

array_view_arg_handler = ArrayViewArgHandler()

# FIXME: configure the blocks
# threadsperblock = 32
# blockspergrid = 128

@cuda.jit(extensions=[array_view_arg_handler])
def swallow(array):
    print("kernel swallow")
    tid = cuda.grid(1)
    size = len(array)
    if tid < size:
        print(array[tid])
    pass

@cuda.jit(extensions=[array_view_arg_handler])
#@cuda.jit
def multiply(array, out, n):
    print("START with kernel multiply...")
    tid = cuda.grid(1)
    print("ask length:")
    size = len(array)
    print("     kernel multiply...", tid, size)
    out[tid] = array[tid] * n
    print(" array[tid]", array[tid])
    print(" out[tid], array[tid]", out[tid], array[tid])
    print("... done with kernel multiply")


# def test_numpy_array_1d():
#    nparray = np.array([10, 1, 2, 3], dtype=int)
#    swallow[1, 4](nparray)
#    print("Done swallow", nparray)


# def test_array_view_1d():
#    akarray = ak.Array([10, 1, 2, 3])
    #assert(isinstance(akarray.layout.nplike, ak.nplikes.Cupy))
#    array_view = ak._connect.numba.arrayview.ArrayView.fromarray(akarray)
#    swallow[1, 4](array_view)
#    print("Done swallow array view", akarray)


#def test_numpy_mult():
#    nparray = np.array([10, 1, 2, 3], dtype=int)
#    out = np.zeros(4, dtype=int)
#    multiply[1, 4](nparray, out, 3)
#    assert ak.to_list(nparray * 3 + 9) == ak.to_list(out)
#    print("Multiply numpy array", nparray, out)


def test_array_multiply_numba():
    akarray = ak.Array([0, 1, 2, 3]) ###, backend="cuda")
    out = np.zeros(4, dtype=np.int64)

    @numba.njit(debug=True)
    def mul(x, o, n):
        print("START mul jitted function--->>>")
        for i in range(len(x)):
            o[i] = x[i] * n
        print("<--- END.")

    mul(akarray, out, 3)
    print("NUMBA jit multiply", out)

def test_array_multiply():
    akarray = ak.Array([0, 1, 2, 3], backend="cuda")
    #????? dev_array1 = ak.to_backend(akarray, "cuda") ### cuda.to_device(arr)
    print_mempool_stats(1)

    #dev_array2 = ak.to_backend(akarray, "cuda")
    #print_mempool_stats(2)

    #dev_array3 = ak.to_backend(akarray, "cuda")
    #print_mempool_stats(3)

    #dev_array4 = ak.to_backend(akarray, "cuda")
    #print_mempool_stats(4)

    arr = np.zeros(4, dtype=np.int64)
    d_arr =  cuda.device_array_like(arr)
    print("CALL kernel multiply...")
    multiply[1, 4](akarray, d_arr, 3)
    print("...done.")
    cuda.synchronize()
    result_array = d_arr.copy_to_host()
    print("Multiply awkward array", akarray, result_array)

    print("AGAIN CALL kernel multiply...")
    multiply[1, 4](akarray, d_arr, 3)
    print("...done.")
    cuda.synchronize()
    result_array = d_arr.copy_to_host()
    print("Multiply awkward array", akarray, result_array)

#    print("ONCE AGAIN CALL kernel multiply...")
#    akarray_cpu = ak.Array([0.0, 1.1, 2.2, 3.3], backend="cpu")

#    multiply[1, 4](akarray_cpu, d_arr, 3)
#    print("...done.")
#    cuda.synchronize()
#    result_array = d_arr.copy_to_host()
#    print("Multiply awkward array", akarray, result_array)

#def test_array_view_multiply():
#    akarray = ak.Array([10, 1, 2, 3], backend="cuda")
#    array_view = ak._connect.numba.arrayview.ArrayView.fromarray(akarray)
#    out = np.zeros(4, dtype=int)
#    d_arr1 = cuda.to_device(out)
#    print_mempool_stats(1)

#    d_arr2 = cuda.to_device(out)
#    print_mempool_stats(2)

#    d_arr3 = cuda.to_device(out)
#    print_mempool_stats(3)

#    d_arr4 = cuda.to_device(out)
#    print_mempool_stats(4)

#    multiply[1, 4](array_view, d_arr1, 3)
#    cuda.synchronize()
#    result_array = d_arr1.copy_to_host()
#    print("Multiply awkward array view", akarray, result_array)

#from numba import cuda

# Define a kernel that is compiled for CUDA
#@cuda.jit
#def vector_add(r, x, y):
#    start = cuda.grid(1)
#    step = cuda.gridsize(1)
#    stop = len(r)
#    for i in range(start, stop, step):
#        r[i] = x[i] + y[i]

#def test_vector():
    
    # Allocate some arrays on the device and copy data
#    N = 2 ** 10
#    x = cuda.to_device(np.arange(N))
#    y = cuda.to_device(np.arange(N) * 2)
#    r = cuda.device_array_like(x)

    # Configure and launch kernel
#    block_dim = 256
#    grid_dim = (len(x) // block_dim) + 1
#    vector_add[grid_dim, block_dim](r, x, y)

    # Copy result back from the device
#    result = r.copy_to_host()
#    print(result)
    
