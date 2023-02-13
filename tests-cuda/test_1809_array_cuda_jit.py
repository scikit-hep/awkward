# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import pytest  # noqa: F401

import awkward as ak  # noqa: F401

numba = pytest.importorskip("numba")

from numba import config, cuda, types  # noqa: F401, E402
from numba.core.typing.typeof import typeof, typeof_impl  # noqa: F401, E402

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


def lower_const_array_as_const_view(array):
    array_view = ak._connect.numba.arrayview.ArrayView.fromarray(array)

    start = id(array_view.start)
    stop = id(array_view.stop)
    pos = id(array_view.pos)
    arrayptrs = id(array_view.lookup.arrayptrs)
    pylookup = id(array_view.lookup)

    return (start, stop, pos, arrayptrs, pylookup)


def lower_as_const_view(array_view):

    start = id(array_view.start)
    stop = id(array_view.stop)
    pos = id(array_view.pos)
    arrayptrs = id(array_view.lookup.arrayptrs)
    pylookup = id(array_view.lookup)

    return (start, stop, pos, arrayptrs, pylookup)


class ArrayViewArgHandler:
    def prepare_args(self, ty, val, stream, retr):
        print("ArrayViewArgHandler::prepare_args", repr(val), type(val))
        if isinstance(val, ak.Array):
            print("val is ak.Array")
            # Use uint64 for start, stop, pos, the array pointers value and the pylookup value
            tys = types.UniTuple(types.uint64, 5)

            # ... Copy data to device if necessary...
            # use similar to: wrap_arg(val).to_device(retr, stream) where
            #
            # def wrap_arg(value, default=InOut):
            #    return value if isinstance(value, ArgHint) else default(value)

            dev = cuda.current_context().device
            print(dev)
            dev_array = ak.to_backend(val, "cuda")  # check if it does copy to device?
            print(
                ">>>>>>>> COPY TO DEVICE! >>>>>>>",
                dev_array,
                repr(dev_array),
                type(dev_array),
            )
            #            ary = np.arange(10)
            #            d_ary = cuda.to_device(ary)
            #            print(d_ary)
            print(dev)

            start, stop, pos, arrayptrs, pylookup = lower_const_array_as_const_view(val)

            # Retrieve data back to host
            def retrieve():
                # Copy dev_array back to host somehow
                hary = ak.to_backend(dev_array, "cpu")  ########d_ary.copy_to_host()
                print("retrieve", hary, repr(hary), type(hary))
                return hary  ###ak.to_backend(dev_array, "cpu")

            # Append retrieve function if necessary
            retr.append(retrieve)

            return tys, (start, stop, pos, arrayptrs, pylookup)

            # print("High level ak.Array to_numpy data pointer is", ak.to_numpy(val).ctypes.data)
            # A pointer to the memory area of the array as a Python integer.
            # return types.uint64, val.layout._data.ctypes.data
            #
            # return types.uint64, ak._connect.numba.arrayview.ArrayView.fromarray(val).lookup.arrayptrs.ctypes.data

        elif isinstance(val, ak._connect.numba.arrayview.ArrayView):
            print("ak.ArrayView")

            tys = types.UniTuple(types.uint64, 5)

            start, stop, pos, arrayptrs, pylookup = lower_as_const_view(val)
            print("Got", start, stop, pos, arrayptrs, pylookup)

            return tys, (start, stop, pos, arrayptrs, pylookup)
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
def multiply(array, out, n):
    tid = cuda.grid(1)
    size = len(array)
    if tid < size:
        out[tid] = n * array[tid]


@cuda.jit(device=True)
def mul(array, out, n):
    tid = cuda.grid(1)
    size = len(array)
    if tid < size:
        out[tid] = n * array[tid]
    return ak.Array(out)


import math  # noqa: F401 
# Note that for the CUDA target, we need to use the scalar functions from the math module, not NumPy


@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y  # This is Python, so let's return a tuple


from numba import vectorize   # noqa: F401


@vectorize(["float32(float32, float32, float32, float32)"], target="cuda")
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def test_numpy_array_1d():
    nparray = np.array([10, 1, 2, 3], dtype=int)
    swallow[1, 1](nparray)
    print("Swallow", nparray)


def test_array_view_1d():
    akarray = ak.Array([10, 1, 2, 3])
    array_view = ak._connect.numba.arrayview.ArrayView.fromarray(akarray)
    swallow[1, 1](array_view)
    print("Swallow array view", akarray)


def test_numpy_mult():
    nparray = np.array([10, 1, 2, 3], dtype=int)
    out = np.zeros(4, dtype=int)
    multiply[1, 4](nparray, out, 3)
    assert ak.to_list(nparray * 3) == ak.to_list(out)
    print("Multiply numpy array", nparray, out)


def test_polar_distance_gpu():
    n = 1000000
    rho1 = ak.Array(np.random.uniform(0.5, 1.5, size=n).astype(np.float32))
    print(repr(rho1), type(rho1))
    theta1 = ak.Array(np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32))
    print(theta1)
    rho2 = ak.Array(np.random.uniform(0.5, 1.5, size=n).astype(np.float32))
    theta2 = ak.Array(np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32))

    out = polar_distance(rho1, theta1, rho2, theta2)
    print(out)


# def test_array_mul():
#    akarray = ak.Array([0, 1, 2, 3])
#    out = np.zeros(4, dtype=int)
###arr = np.zeros(4, dtype=int) ###np.arange(1000)
###d_arr = cuda.to_device(arr)

#    result_array = mul(akarray, out, 3)
#    print("Multiply (device func) awkward array", akarray, result_array)


def test_array_multiply():
    akarray = ak.Array([0, 1, 2, 3])
    ###out = np.zeros(4, dtype=int)
    arr = np.zeros(4, dtype=int)  ###np.arange(1000)
    d_arr = cuda.to_device(arr)

    multiply[1, 4](akarray, d_arr, 3)
    result_array = d_arr.copy_to_host()
    print("Multiply awkward array", akarray, result_array)


def test_array_view_multiply():
    akarray = ak.Array([10, 1, 2, 3])
    array_view = ak._connect.numba.arrayview.ArrayView.fromarray(akarray)
    out = np.zeros(4, dtype=int)
    d_arr = cuda.to_device(out)
    multiply[1, 4](array_view, out, 3)
    result_array = d_arr.copy_to_host()
    print("Multiply awkward array view", akarray, result_array)


# def test_list_array_multipy():
#    akarray = ak.Array([[0, 1], [2], [3, 4,5]])
#    multiply[1, 1](akarray, 3)
#    print("Multiply list array", akarray)


def test_as_array_1d():
    akarray = ak.Array([0, 1, 2, 3])
    print("1. Swallaw ndarray", akarray)
    swallow[1, 1](np.asarray(akarray))
    print("2. Swallaw ndarray", akarray)
    swallow[1, 1](np.asarray(akarray))
    print("3. Swallaw ndarray", akarray)
    swallow[1, 1](np.asarray(akarray))
    print("4. Swallaw ndarray", akarray)
    swallow[1, 1](np.asarray(akarray))

    print("5. Swallaw ak.Array", akarray)
    swallow[1, 1](akarray)
    print("6. Swallaw ak.Array", akarray)
    swallow[1, 1](akarray)
    print("7. Swallaw ak.Array", akarray)
    swallow[1, 1](akarray)
    print("8. Swallaw ak.Array", akarray)
    swallow[1, 1](akarray)


def test_NumpyArrayType_array():
    akarray = ak.Array([0, 1, 2, 3])
    swallow[1, 1](akarray)


def test_ListArrayType_array():
    akarray0 = ak.Array([0, 1, 2, 3])
    swallow[1, 1](akarray0)
    akarray1 = ak.Array([[0, 1], [2], [3, 4, 5]])
    swallow[1, 1](akarray1)
