# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

from cuda.compute import (
    ZipIterator,
    gpu_struct,
    reduce_into,
)

import awkward as ak
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._reducers import Reducer
from awkward._typing import Any as AnyType
from awkward._typing import Self, TypeVar

np = NumpyMetadata.instance()

DTypeLike = AnyType

_overloads: dict[type[Reducer], type[CudaComputeReducer]] = {}


R = TypeVar("R", bound=Reducer)


def overloads(cls: type[Reducer]):
    def registrar(new_cls: type[R]) -> type[R]:
        _overloads[cls] = new_cls
        return new_cls

    return registrar


class CudaComputeReducer(Reducer):
    @classmethod
    @abstractmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        raise NotImplementedError

    @classmethod
    def _dtype_for_kernel(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        if dtype.kind.upper() == "M":
            return np.dtype(np.int64)
        elif dtype == np.complex128:
            return np.dtype(np.float64)
        elif dtype == np.complex64:
            return np.dtype(np.float32)
        else:
            return dtype


# TODO: change this for a custom one?
def apply_positional_corrections(
    reduced: ak.contents.NumpyArray,
    parents: ak.index.Index | ak.index.ZeroIndex,
    offsets: ak.index.Index | ak.index.EmptyIndex,
    starts: ak.index.Index,
    shifts: ak.index.Index | None,
):
    if shifts is None:
        assert (
            parents.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
        )
        reduced.backend.maybe_kernel_error(
            reduced.backend[
                "awkward_NumpyArray_reduce_adjust_starts_64",
                reduced.dtype.type,
                parents.dtype.type,
                starts.dtype.type,
            ](
                reduced.data,
                reduced.length,
                parents.data,
                starts.data,
            )
        )
        return reduced
    else:
        assert (
            parents.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
            and shifts.nplike is reduced.backend.nplike
        )
        reduced.backend.maybe_kernel_error(
            reduced.backend[
                "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
                reduced.dtype.type,
                parents.dtype.type,
                starts.dtype.type,
                shifts.dtype.type,
            ](
                reduced.data,
                reduced.length,
                parents.data,
                starts.data,
                shifts.data,
            )
        )
        return reduced


def awkward_axis_none_reduce_argmin(array):
    import cupy as cp

    data_dtype = array.dtype
    index_dtype = np.int64
    # initialize the minimum value depending on the dtype
    if data_dtype.kind in "iu":  # int/uint
        max = cp.iinfo(data_dtype).max
    elif data_dtype.kind == "f":  # float
        max = cp.finfo(data_dtype).max
    elif data_dtype.kind == "c":  # complex
        max = complex(cp.finfo(data_dtype).max, cp.finfo(data_dtype).max)
    else:
        raise TypeError("Unsupported dtype to get the minimal value")

    ak_array = gpu_struct(
        {
            "data": data_dtype,
            "local_index": index_dtype,
        }
    )

    if data_dtype.kind == "c":
        # Compare real and then imaginary part for complex numbers (same way as in cpu kernels)
        def reduce_op(a: ak_array, b: ak_array):
            real_a, imag_a = a.data.real, a.data.imag
            real_b, imag_b = b.data.real, b.data.imag
            # we are sorting lexicographically
            if (real_b, imag_b) != (real_a, imag_a):
                return b if (real_b, imag_b) < (real_a, imag_a) else a
            # if the two values are equal, keep the index of the FIRST min value we encounter
            if a.local_index < b.local_index:
                return a
            elif b.local_index < a.local_index:
                return b
            else:
                raise IndexError("Encountered two values with the same index")
    else:
        # Normal numeric comparison
        def reduce_op(a: ak_array, b: ak_array):
            if a.data != b.data:
                return a if a.data < b.data else b
            elif a.local_index != b.local_index:
                # for the equal values always return the one with the lowest index
                return a if a.local_index < b.local_index else b
            else:
                raise IndexError("Encountered two values with the same index")

    n = len(array)
    indices = cp.arange(n, dtype=index_dtype)
    # Combine data and their indices into a single structure
    input_struct = ZipIterator(array, indices)

    result_scalar = cp.empty(1, dtype=ak_array)
    h_init = ak_array(max, -1)
    reduce_into(
        d_in=input_struct,
        d_out=result_scalar,
        op=reduce_op,
        num_items=len(array),
        h_init=h_init,
    )

    # return the index of the maximum value
    return result_scalar.item()[1]


def awkward_axis_none_reduce_argmax(array):
    import cupy as cp

    data_dtype = array.dtype
    index_dtype = np.int64
    # initialize the minimum value depending on the dtype
    if data_dtype.kind in "iu":  # int/uint
        min = cp.iinfo(data_dtype).min
    elif data_dtype.kind == "f":  # float
        min = cp.finfo(data_dtype).min
    elif data_dtype.kind == "c":  # complex
        min = complex(cp.finfo(data_dtype).min, cp.finfo(data_dtype).min)
    else:
        raise TypeError("Unsupported dtype to get the minimal value")

    ak_array = gpu_struct(
        {
            "data": data_dtype,
            "local_index": index_dtype,
        }
    )

    if data_dtype.kind == "c":
        # Compare real and then imaginary part for complex numbers (same way as in cpu kernels)
        def reduce_op(a: ak_array, b: ak_array):
            real_a, imag_a = a.data.real, a.data.imag
            real_b, imag_b = b.data.real, b.data.imag
            # we are sorting lexicographically
            if (real_b, imag_b) != (real_a, imag_a):
                return b if (real_b, imag_b) > (real_a, imag_a) else a
            # if the two values are equal, keep the index of the FIRST max value we encounter
            if a.local_index < b.local_index:
                return a
            elif b.local_index < a.local_index:
                return b
            else:
                raise IndexError("Encountered two values with the same index")
    else:
        # Normal numeric comparison
        def reduce_op(a: ak_array, b: ak_array):
            if a.data != b.data:
                return a if a.data > b.data else b
            elif a.local_index != b.local_index:
                # for the equal values always return the one with the lowest index
                return a if a.local_index < b.local_index else b
            else:
                raise IndexError("Encountered two values with the same index")

    n = len(array)
    indices = cp.arange(n, dtype=index_dtype)
    # Combine data and their indices into a single structure
    input_struct = ZipIterator(array, indices)

    result_scalar = cp.empty(1, dtype=ak_array)
    h_init = ak_array(min, -1)
    reduce_into(
        d_in=input_struct,
        d_out=result_scalar,
        op=reduce_op,
        num_items=len(array),
        h_init=h_init,
    )

    # return the index of the maximum value
    return result_scalar.item()[1]


def get_cuda_compute_reducer(reducer: Reducer) -> Reducer:
    """
    Returns the CUDA-specific reducer if registered,
    otherwise falls back to the original reducer.
    """
    impl = _overloads.get(type(reducer))
    if impl is None:
        return reducer

    return impl.from_kernel_reducer(reducer)
