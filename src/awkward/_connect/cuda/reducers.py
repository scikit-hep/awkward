# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

from cuda.compute import (
    ZipIterator,
    gpu_struct,
    reduce_into,
)

import awkward as ak
from awkward import _reducers
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._reducers import Reducer
from awkward._typing import Any as AnyType
from awkward._typing import Final, Self, TypeVar

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
    def _length_for_kernel(cls, type: DTypeLike, length: ShapeItem) -> ShapeItem:
        # Complex types are implemented as double-wide arrays
        return 2 * length if type in (np.complex128, np.complex64) else length

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

    _use32 = ((ak._util.win or ak._util.bits32) and not ak._util.numpy2) or (
        ak._util.numpy2 and np.intp is np.int32
    )

    @classmethod
    def _promote_integer_rank(cls, given_dtype: DTypeLike) -> DTypeLike:
        if given_dtype in (np.bool_, np.int8, np.int16, np.int32):
            return np.int32 if cls._use32 else np.int64

        elif given_dtype in (np.uint8, np.uint16, np.uint32):
            return np.uint32 if cls._use32 else np.uint64

        else:
            return given_dtype


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


@overloads(_reducers.ArgMin)
class ArgMin(CudaComputeReducer):
    name: Final = "argmin"
    needs_position: Final = True
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.ArgMin)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def _dtype_for_kernel(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        if dtype == np.bool_:
            return np.dtype(np.int8)
        else:
            return super()._dtype_for_kernel(dtype)

    def axis_none_reducer(self):
        return AxisNoneReducerArgMaxMin(self.name)

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)

        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmin_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    parents.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.backend.nplike
            from ._compute import awkward_reduce_argmin

            awkward_reduce_argmin(
                result,
                kernel_array_data,
                parents.data,
                offsets.data,
                parents.length,
                starts.data,
                outlength,
            )

        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        corrected_data = apply_positional_corrections(
            result_array, parents, offsets, starts, shifts
        )
        return corrected_data


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
    reduce_into(input_struct, result_scalar, reduce_op, len(array), h_init)

    # return the index of the maximum value
    return result_scalar.item()[1]


@overloads(_reducers.ArgMax)
class ArgMax(CudaComputeReducer):
    name: Final = "argmax"
    needs_position: Final = True
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.ArgMax)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    @classmethod
    def _dtype_for_kernel(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        if dtype == np.bool_:
            return np.dtype(np.int8)
        else:
            return super()._dtype_for_kernel(dtype)

    def axis_none_reducer(self):
        return AxisNoneReducerArgMaxMin(self.name)

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)

        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmax_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    parents.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.backend.nplike
            from ._compute import awkward_reduce_argmax

            awkward_reduce_argmax(
                result,
                kernel_array_data,
                parents.data,
                offsets.data,
                parents.length,
                starts.data,
                outlength,
            )

        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        corrected_data = apply_positional_corrections(
            result_array, parents, offsets, starts, shifts
        )
        return corrected_data


class AxisNoneReducerArgMaxMin:
    # This will construct shifts which we can use to `apply_positional_corrections` at the end.
    needs_position: Final = True

    def __init__(self, name: str):
        self.name = name

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        _outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(
                f"cannot compute the {self.name} (ak.{self.name}) of {array.dtype!r}"
            )

        nplike = array.backend.nplike
        reduce_fn = globals()[f"awkward_axis_none_reduce_{self.name}"]
        result_scalar = reduce_fn(array.data)

        # is this line needed?
        result_array = nplike._module.asarray(result_scalar).reshape((1,))

        result_array = ak.contents.NumpyArray(result_array, backend=array.backend)
        # We need to do this, as for argmax we care for the position of the max element and not only the data itself.
        corrected_data = apply_positional_corrections(
            result_array, parents, offsets, starts, shifts
        )

        return corrected_data


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
    reduce_into(input_struct, result_scalar, reduce_op, len(array), h_init)

    # return the index of the maximum value
    return result_scalar.item()[1]


@overloads(_reducers.Count)
class Count(CudaComputeReducer):
    name: Final = "count"
    needs_position: Final = False
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Count)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        result = array.backend.nplike.empty(outlength, dtype=np.int64)

        assert parents.nplike is array.backend.nplike
        from ._compute import awkward_reduce_count_64

        awkward_reduce_count_64(
            result,
            parents.data,
            parents.length,
            outlength,
        )

        return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.CountNonzero)
class CountNonzero(CudaComputeReducer):
    name: Final = "count_nonzero"
    needs_position: Final = False
    preferred_dtype: Final = np.int64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.CountNonzero)
        return cls()

    @classmethod
    def _return_dtype(cls, given_dtype):
        return np.int64

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)

        assert parents.nplike is array.backend.nplike
        if np.issubdtype(array.dtype, np.complexfloating):
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_countnonzero_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            from ._compute import awkward_reduce_countnonzero

            awkward_reduce_countnonzero(
                result,
                kernel_array_data,
                parents.data,
                parents.length,
                outlength,
            )

        return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.Sum)
class Sum(CudaComputeReducer):
    name: Final = "sum"
    needs_position: Final = False
    preferred_dtype: Final = np.float64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Sum)
        return cls()

    def axis_none_reducer(self):
        return AxisNoneCudaSum()

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")

        # Boolean kernels are special; the result is _not_ a boolean
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=self._promote_integer_rank(np.bool_),
            )
            assert parents.nplike is array.backend.nplike
            if result.dtype in (np.int64, np.uint64, np.int32, np.uint32):
                from ._compute import awkward_reduce_sum_int32_bool_64

                awkward_reduce_sum_int32_bool_64(
                    result,
                    array.data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                )
            else:
                raise NotImplementedError
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=self._promote_integer_rank(kernel_array_data.dtype),
            )
            assert parents.nplike is array.backend.nplike
            if array.dtype.type in (np.complex128, np.complex64):
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        offsets.data,
                        parents.length,
                        outlength,
                    )
                )
            else:
                from ._compute import awkward_reduce_sum

                awkward_reduce_sum(
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                )
            return ak.contents.NumpyArray(
                result.view(self._promote_integer_rank(array.dtype)),
                backend=array.backend,
            )


class AxisNoneCudaSum:
    name: Final = "sum"
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        _parents: ak.index.Index | ak.index.ZeroIndex,
        _offsets: ak.index.Index | ak.index.EmptyIndex,
        _starts: ak.index.Index,
        _shifts: ak.index.Index | None,
        _outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")

        nplike = array.backend.nplike
        reduce_fn = getattr(nplike, self.name)

        result_scalar = reduce_fn(array.data, axis=None)
        result_array = array.backend.nplike.reshape(
            array.backend.nplike._module.asarray(result_scalar), (1,)
        )
        return ak.contents.NumpyArray(result_array, backend=array.backend)


@overloads(_reducers.Prod)
class Prod(CudaComputeReducer):
    name: Final = "prod"
    needs_position: Final = False
    preferred_dtype: Final = np.float64

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Prod)
        return cls()

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind.upper() == "M":
            raise ValueError(f"cannot compute the product (ak.prod) of {array.dtype!r}")

        # Boolean kernels are special; the result is _not_ a boolean
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(
                outlength,
                # This kernel, unlike sum, returns bools!
                dtype=np.bool_,
            )
            assert parents.nplike is array.backend.nplike
            from ._compute import awkward_reduce_prod_bool

            awkward_reduce_prod_bool(
                result,
                array.data,
                parents.data,
                offsets.data,
                parents.length,
                outlength,
            )
            return ak.contents.NumpyArray(
                array.backend.nplike.astype(
                    result, dtype=self._promote_integer_rank(array.dtype)
                ),
                backend=array.backend,
            )
        else:
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=self._promote_integer_rank(kernel_array_data.dtype),
            )
            assert parents.nplike is array.backend.nplike
            if array.dtype.type in (np.complex128, np.complex64):
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_prod_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        offsets.data,
                        parents.length,
                        outlength,
                    )
                )
            else:
                from ._compute import awkward_reduce_prod

                awkward_reduce_prod(
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                )
            return ak.contents.NumpyArray(
                result.view(self._promote_integer_rank(array.dtype)),
                backend=array.backend,
            )


@overloads(_reducers.Any)
class Any(CudaComputeReducer):
    name: Final = "any"
    needs_position: Final = False
    preferred_dtype: Final = np.bool_

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Any)
        return cls()

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)

        assert parents.nplike is array.backend.nplike
        if array.dtype.type in (np.complex128, np.complex64):
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_sum_bool_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    parents.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            from ._compute import awkward_reduce_sum_bool

            awkward_reduce_sum_bool(
                result,
                kernel_array_data,
                parents.data,
                offsets.data,
                parents.length,
                outlength,
            )
        return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.All)
class All(CudaComputeReducer):
    name: Final = "all"
    needs_position: Final = False
    preferred_dtype: Final = np.bool_

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.All)
        return cls()

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)

        assert parents.nplike is array.backend.nplike
        if array.dtype.type in (np.complex128, np.complex64):
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    parents.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            from ._compute import awkward_reduce_prod_bool

            awkward_reduce_prod_bool(
                result,
                kernel_array_data,
                parents.data,
                offsets.data,
                parents.length,
                outlength,
            )
        return ak.contents.NumpyArray(result, backend=array.backend)


@overloads(_reducers.Min)
class Min(CudaComputeReducer):
    name: Final = "min"
    needs_position: Final = False
    preferred_dtype: Final = np.float64

    def __init__(self, initial: float | None):
        self._initial = initial

    @property
    def initial(self) -> float | None:
        return self._initial

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Min)
        return cls(reducer.initial)

    def axis_none_reducer(self):
        return AxisNoneCudaMin(self._initial)

    def _identity_for(self, dtype: DTypeLike | None):
        dtype = np.dtype(dtype)
        assert dtype.kind.upper() != "M", (
            "datetime64/timedelta64 should be converted to int64 before reduction"
        )
        if self._initial is None:
            if dtype in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(dtype).max
            else:
                return np.inf
        return self._initial

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(outlength, dtype=np.bool_)
            assert parents.nplike is array.backend.nplike
            from ._compute import awkward_reduce_prod_bool

            awkward_reduce_prod_bool(
                result,
                array.data,
                parents.data,
                offsets.data,
                parents.length,
                outlength,
            )
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=kernel_array_data.dtype,
            )
            assert parents.nplike is array.backend.nplike
            if array.dtype.type in (np.complex128, np.complex64):
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_min_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        offsets.data,
                        parents.length,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            else:
                from ._compute import awkward_reduce_min

                awkward_reduce_min(
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                    self._identity_for(result.dtype),
                )
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )


class AxisNoneCudaMin(Min):
    def apply(
        self,
        array: ak.contents.NumpyArray,
        _parents: ak.index.Index | ak.index.ZeroIndex,
        _offsets: ak.index.Index | ak.index.EmptyIndex,
        _starts: ak.index.Index,
        _shifts: ak.index.Index | None,
        _outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)

        nplike = array.backend.nplike
        data = array.data

        if data.size == 0:
            val = (
                self.initial
                if self.initial is not None
                else self._identity_for(data.dtype)
            )
            result_scalar = nplike.asarray(val, dtype=data.dtype)
        else:
            result_scalar = nplike.min(data)
            if self.initial is not None:
                initial_val = nplike.asarray(self.initial, dtype=data.dtype)
                result_scalar = nplike.minimum(result_scalar, initial_val)

        result_array = nplike.reshape(nplike.asarray(result_scalar), (1,))

        return ak.contents.NumpyArray(result_array, backend=array.backend)


@overloads(_reducers.Max)
class Max(CudaComputeReducer):
    name: Final = "max"
    needs_position: Final = False
    preferred_dtype: Final = np.float64

    def __init__(self, initial: float | None):
        self._initial = initial

    @property
    def initial(self) -> float | None:
        return self._initial

    @classmethod
    def from_kernel_reducer(cls, reducer: Reducer) -> Self:
        assert isinstance(reducer, _reducers.Max)
        return cls(reducer.initial)

    def axis_none_reducer(self):
        return AxisNoneCudaMax(self._initial)

    def _identity_for(self, dtype: DTypeLike | None):
        dtype = np.dtype(dtype)
        assert dtype.kind.upper() != "M", (
            "datetime64/timedelta64 should be converted to int64 before reduction"
        )
        if self._initial is None:
            if dtype in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(dtype).min
            else:
                return -np.inf
        return self._initial

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index | ak.index.ZeroIndex,
        offsets: ak.index.Index | ak.index.EmptyIndex,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(outlength, dtype=np.bool_)
            assert parents.nplike is array.backend.nplike
            from ._compute import awkward_reduce_sum_bool

            awkward_reduce_sum_bool(
                result,
                array.data,
                parents.data,
                offsets.data,
                parents.length,
                outlength,
            )
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=kernel_array_data.dtype,
            )
            assert parents.nplike is array.backend.nplike
            if array.dtype.type in (np.complex128, np.complex64):
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_max_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        offsets.data,
                        parents.length,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            else:
                from ._compute import awkward_reduce_max

                awkward_reduce_max(
                    result,
                    kernel_array_data,
                    parents.data,
                    offsets.data,
                    parents.length,
                    outlength,
                    self._identity_for(result.dtype),
                )
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )


class AxisNoneCudaMax(Max):
    def apply(
        self,
        array: ak.contents.NumpyArray,
        _parents: ak.index.Index | ak.index.ZeroIndex,
        _offsets: ak.index.Index | ak.index.EmptyIndex,
        _starts: ak.index.Index,
        _shifts: ak.index.Index | None,
        _outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)

        nplike = array.backend.nplike
        data = array.data

        if data.size == 0:
            val = (
                self.initial
                if self.initial is not None
                else self._identity_for(data.dtype)
            )
            result_scalar = nplike.asarray(val, dtype=data.dtype)
        else:
            result_scalar = nplike.max(data)
            if self.initial is not None:
                initial_val = nplike.asarray(self.initial, dtype=data.dtype)
                result_scalar = nplike.maximum(result_scalar, initial_val)

        result_array = nplike.reshape(nplike.asarray(result_scalar), (1,))

        return ak.contents.NumpyArray(result_array, backend=array.backend)


def get_cuda_compute_reducer(reducer: Reducer) -> Reducer:
    """
    Returns the CUDA-specific reducer if registered,
    otherwise falls back to the original reducer.
    """
    impl = _overloads.get(type(reducer))
    if impl is None:
        return reducer

    return impl.from_kernel_reducer(reducer)
