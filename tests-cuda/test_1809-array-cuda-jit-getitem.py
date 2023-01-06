# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401
import ctypes  # noqa: F401

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
    print("print_mempool_stats: -------------------------")
    print("print_mempool_stats:", idx, ": mempool.used_bytes", mempool.used_bytes())
    print("print_mempool_stats:", idx, ": mempool.total_bytes", mempool.total_bytes())
    print("print_mempool_stats:", idx, ": pinned_mempool.n_free_blocks", pinned_mempool.n_free_blocks())

class ArrayViewArgHandler:
    print("class ArrayViewHandler: access statistics of the memory pools:")
    print_mempool_stats(0)

    def prepare_args(self, ty, val, stream, retr):
        if isinstance(val, int):
            result = format(val, 'x')
            print("ArrayViewArgHandler::prepare_args line 93:", result)
        if isinstance(val, ak.Array):
            print("ArrayViewArgHandler::prepare_args line 95: ______________________________________________")
            print("ArrayViewArgHandler::prepare_args line 96: ArrayViewArgHandler::prepare_args for ak.Array")
            print("ArrayViewArgHandler::prepare_args line 97: ----------------------------------------------")
            #print(ty, dir(ty), ty.type)
            #print(val, dir(val), val._numbaview, dir(val._numbaview))

            if isinstance(val.layout.nplike, ak.nplikes.Cupy):

                # Use uint64 for start, stop, pos, the array pointers value and the pylookup value
                tys = types.UniTuple(types.uint64, 5)
                print("ArrayViewArgHandler::prepare_args line 105: Already has an ArrayView:", val._numbaview.lookup, val._numbaview.pos, val._numbaview.start, val._numbaview.stop)

                dev = cuda.current_context().device
                print("ArrayViewArgHandler::prepare_args line 108:", dev)

                # access statistics of these memory pools.
                print_mempool_stats(1)
 
                start = id(val._numbaview.start)
                stop = id(val._numbaview.stop)
                pos = id(val._numbaview.pos)
                arrayptrs = val._numbaview.lookup.arrayptrs
                pylookup = id(val._numbaview.lookup)

                result_ptr = format(arrayptrs.item(), 'x')
                print("ArrayViewArgHandler::prepare_args line 119: about to return from prepare args and arrayptrs is", result_ptr, arrayptrs.item())
                print("ArrayViewArgHandler::prepare_args line 120:", arrayptrs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)).contents)
                
                return tys, (start, stop, pos, arrayptrs.item(), pylookup)
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

# threads = 64
# blocks = 64
# nthreads = blocks * threads


@cuda.jit(debug=True, opt=False, extensions=[array_view_arg_handler])
def multiply(array, n, out):
    tid = cuda.grid(1)
    print("     kernel multiply for tid...", tid)
    out[tid] = array[tid] * n


def test_array_multiply():

    if numba.cuda.is_available():
        print("test_array_multiply line 150: CUDA GPU is available!")
    else:
        print("test_array_multiply line 152: NO CUDA GPU...")
        
    numba.cuda.detect()

    print("test_array_multiply line 156: create an ak.Array with cuda backend:")
    akarray = ak.Array([0, 1, 2, 3], backend="cuda")
    print("test_array_multiply line 158: access statistics of the memory pools:")
    print_mempool_stats(1)

    print("test_array_multiply line 161: allocate the result:")
    nthreads = 4
    results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))
    print("test_array_multiply line 164: access statistics of the memory pools:")
    print_mempool_stats(2)

    print("test_array_multiply line 167: CALL kernel multiply...")
    multiply[1, 4](akarray, 3, results)
    print("test_array_multiply line 169: ...done.")
    cuda.synchronize()
    host_results = results.copy_to_host()

    print("test_array_multiply line 173: multiplied Awkward Array", host_results)
