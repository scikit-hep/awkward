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
        if isinstance(val, int):
            result = format(val, 'x')
            print(result)
        if isinstance(val, ak.Array):
            print(ty, dir(ty), ty.type)
            print(val, dir(val), val._numbaview, dir(val._numbaview))

            if isinstance(val.layout.nplike, ak.nplikes.Cupy):

                # Use uint64 for start, stop, pos, the array pointers value and the pylookup value
                tys = types.UniTuple(types.uint64, 5)
                print("Already has an ArrayView:", val._numbaview.lookup, val._numbaview.pos, val._numbaview.start, val._numbaview.stop)

                dev = cuda.current_context().device
                print(dev)

                # access statistics of these memory pools.
                print_mempool_stats(1)

                start = val._numbaview.start
                stop = val._numbaview.stop
                pos = val._numbaview.pos
                arrayptrs = val._numbaview.lookup.arrayptrs.data.ptr
                pylookup = id(val._numbaview.lookup)

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


def test_array_multiply():
    akarray = ak.Array([0, 1, 2, 3], backend="cuda")
    print_mempool_stats(1)

    arr = np.zeros(4, dtype=np.int64)
    d_arr =  cuda.device_array_like(arr)
    print("CALL kernel multiply...")
    multiply[1, 4](akarray, d_arr, 3)
    print("...done.")
    cuda.synchronize()
    result_array = d_arr.copy_to_host()
    print("Multiply awkward array", result_array)

#include <cstdint>

struct A {
  A();

  ~A();

  int *p_;

  intptr_t getDevicePointer() const;
};

#include <arrayPointer.hpp>
#include <cstdint>

A::A() {
  cudaMalloc((void**)&p_, 10 * sizeof(int));
  cudaMemset((void*)p_, 0, 10 * sizeof(int));
  cudaDeviceSynchronize();
}

A::~A() {
  cudaFree((void*)p_);
}

intptr_t A::getDevicePointer() const {
  return reinterpret_cast<intptr_t>(p_);
}

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(arrayPointer_pybind, m) {
  m.doc() = "Test passing CUDA device memory pointer to cupy array";

  py::class_<A>(m, "A")
    .def(py::init())
    .def("getDevicePointer", &A::getDevicePointer)
    ;

}


import cupy as cp
import arrayPointer_pybind
from cupy.cuda.memory import MemoryPointer, UnownedMemory

# span is the total number of bytes.
# dtype is int32
def pDevice2CuPyArray(pDevice : int, span : int, dtype : type, owner=None):
    sizeByte = span * cp.dtype(dtype).itemsize
    mem = UnownedMemory(pDevice, sizeByte, owner)
    memptr = MemoryPointer(mem, 0)
    return cp.ndarray(span, dtype=dtype, memptr=memptr)


a = arrayPointer_pybind.A()

pDevice = a.getDevicePointer()
arrDevice = pDevice2CuPyArray(pDevice, 10, cp.int32)
