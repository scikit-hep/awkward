# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod

import awkward as ak
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._typing import Any as AnyType
from awkward._typing import Final, Protocol

np = NumpyMetadata.instance()
numpy = Numpy.instance()

DTypeLike = AnyType


class Reducer(Protocol):
    name: str

    # Does the output correspond to array positions?
    @property
    @abstractmethod
    def needs_position(self) -> bool: ...

    @property
    @abstractmethod
    def preferred_dtype(self) -> DTypeLike: ...

    @classmethod
    def highlevel_function(cls):
        return getattr(ak.operations, cls.name)

    @abstractmethod
    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray: ...


class KernelReducer(Reducer):
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

    _use32 = (ak._util.win or ak._util.bits32) and not ak._util.numpy2

    @classmethod
    def _promote_integer_rank(cls, given_dtype: DTypeLike) -> DTypeLike:
        if given_dtype in (np.bool_, np.int8, np.int16, np.int32):
            return np.int32 if cls._use32 else np.int64

        elif given_dtype in (np.uint8, np.uint16, np.uint32):
            return np.uint32 if cls._use32 else np.uint64

        else:
            return given_dtype


def apply_positional_corrections(
    reduced: ak.contents.NumpyArray,
    parents: ak.index.Index,
    starts: ak.index.Index,
    shifts: ak.index.Index | None,
):
    if shifts is None:
        assert (
            parents.nplike is reduced.backend.index_nplike
            and starts.nplike is reduced.backend.index_nplike
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
    else:
        assert (
            parents.nplike is reduced.backend.index_nplike
            and starts.nplike is reduced.backend.index_nplike
            and shifts.nplike is reduced.backend.index_nplike
        )
        reduced.backend.maybe_kernel_error(
            reduced._backend[
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


class ArgMin(KernelReducer):
    name: Final = "argmin"
    preferred_dtype: Final = np.int64
    needs_position: Final = True

    @classmethod
    def _dtype_for_kernel(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        if dtype == np.bool_:
            return np.dtype(np.int8)
        else:
            return super()._dtype_for_kernel(dtype)

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmin_complex",
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
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmin",
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
        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        apply_positional_corrections(result_array, parents, starts, shifts)
        return result_array


class ArgMax(KernelReducer):
    name: Final = "argmax"
    preferred_dtype: Final = np.int64
    needs_position: Final = True

    @classmethod
    def _dtype_for_kernel(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        if dtype == np.bool_:
            return np.dtype(np.int8)
        else:
            return super()._dtype_for_kernel(dtype)

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmax_complex",
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
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmax",
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
        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        apply_positional_corrections(result_array, parents, starts, shifts)
        return result_array


class Count(KernelReducer):
    name: Final = "count"
    preferred_dtype: Final = np.int64
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        assert parents.nplike is array.backend.index_nplike
        array.backend.maybe_kernel_error(
            array.backend[
                "awkward_reduce_count_64", result.dtype.type, parents.dtype.type
            ](
                result,
                parents.data,
                parents.length,
                outlength,
            )
        )
        return ak.contents.NumpyArray(result, backend=array.backend)


class CountNonzero(KernelReducer):
    name: Final = "count_nonzero"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))

        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if np.issubdtype(array.dtype, np.complexfloating):
            assert parents.nplike is array.backend.index_nplike
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
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_countnonzero",
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
        return ak.contents.NumpyArray(result, backend=array.backend)


class Sum(KernelReducer):
    name: Final = "sum"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
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
            if result.dtype in (np.int64, np.uint64):
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum_int64_bool_64",
                        np.int64,
                        array.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        array.data,
                        parents.data,
                        parents.length,
                        outlength,
                    )
                )
            elif result.dtype in (np.int32, np.uint32):
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum_int32_bool_64",
                        np.int32,
                        array.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        array.data,
                        parents.data,
                        parents.length,
                        outlength,
                    )
                )
            else:
                raise NotImplementedError
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            # View array data in kernel-supported dtype
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=self._promote_integer_rank(kernel_array_data.dtype),
            )
            if array.dtype.type in (np.complex128, np.complex64):
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum_complex",
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
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum",
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

            return ak.contents.NumpyArray(
                result.view(self._promote_integer_rank(array.dtype)),
                backend=array.backend,
            )


class Prod(KernelReducer):
    name: Final = "prod"
    preferred_dtype: Final = np.int64
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
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
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
            return ak.contents.NumpyArray(
                array.backend.nplike.astype(
                    result, dtype=self._promote_integer_rank(array.dtype)
                ),
                backend=array.backend,
            )
        else:
            # View array data in kernel-supported dtype
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=self._promote_integer_rank(kernel_array_data.dtype),
            )
            if array.dtype.type in (np.complex128, np.complex64):
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_prod_complex",
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
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_prod",
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

            return ak.contents.NumpyArray(
                result.view(self._promote_integer_rank(array.dtype)),
                backend=array.backend,
            )


class Any(KernelReducer):
    name: Final = "any"
    preferred_dtype: Final = np.bool_
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)

        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_sum_bool_complex",
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
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_sum_bool",
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
        return ak.contents.NumpyArray(result, backend=array.backend)


class All(KernelReducer):
    name: Final = "all"
    preferred_dtype: Final = np.bool_
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        # View array data in kernel-supported dtype
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)

        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool_complex",
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
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool",
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
        return ak.contents.NumpyArray(result, backend=array.backend)


class Min(KernelReducer):
    name: Final = "min"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, initial: float | None):
        self._initial = initial

    @property
    def initial(self) -> float | None:
        return self._initial

    def _identity_for(self, dtype: DTypeLike | None) -> float:
        dtype = np.dtype(dtype)

        assert (
            dtype.kind.upper() != "M"
        ), "datetime64/timedelta64 should be converted to int64 before reduction"
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
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(outlength, dtype=np.bool_)
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            # View array data in kernel-supported dtype
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=kernel_array_data.dtype,
            )
            if array.dtype.type in (np.complex128, np.complex64):
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_min_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        parents.length,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            else:
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_min",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        parents.length,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )


class Max(KernelReducer):
    name: Final = "max"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    def _identity_for(self, dtype: DTypeLike | None):
        dtype = np.dtype(dtype)

        assert (
            dtype.kind.upper() != "M"
        ), "datetime64/timedelta64 should be converted to int64 before reduction"
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
        parents: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(outlength, dtype=np.bool_)
            assert parents.nplike is array.backend.index_nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_sum_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            # View array data in kernel-supported dtype
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=kernel_array_data.dtype,
            )
            if array.dtype.type in (np.complex128, np.complex64):
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_max_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        parents.length,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            else:
                assert parents.nplike is array.backend.index_nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_max",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        parents.data,
                        parents.length,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )
