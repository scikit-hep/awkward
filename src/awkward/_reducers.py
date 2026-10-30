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

    def axis_none_reducer(self) -> Reducer | None:
        """A specialized version for axis=None reductions, or None if there is none."""
        return None

    @abstractmethod
    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray: ...


class KernelReducer(Reducer):
    @classmethod
    def _length_for_kernel(cls, type: DTypeLike, length: ShapeItem) -> ShapeItem:
        # Complex types are implemented as double-wide arrays
        return 2 * length if np.dtype(type).kind == "c" else length

    @classmethod
    def _dtype_for_kernel(cls, dtype: DTypeLike) -> DTypeLike:
        dtype = np.dtype(dtype)
        if dtype.kind.upper() == "M":
            return np.dtype(np.int64)
        elif dtype == np.complex128:
            return np.dtype(np.float64)
        elif dtype == np.complex64:
            return np.dtype(np.float32)
        elif dtype.kind == "c":
            return np.dtype("longdouble")
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


def apply_positional_corrections(
    reduced: ak.contents.NumpyArray,
    offsets: ak.index.Index,
    starts: ak.index.Index,
    shifts: ak.index.Index | None,
):
    if shifts is None:
        assert (
            offsets.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
        )
        reduced.backend.maybe_kernel_error(
            reduced.backend[
                "awkward_NumpyArray_reduce_adjust_starts_64",
                reduced.dtype.type,
                offsets.dtype.type,
                starts.dtype.type,
            ](
                reduced.data,
                reduced.length,
                offsets.data,
                starts.data,
            )
        )
    else:
        assert (
            offsets.nplike is reduced.backend.nplike
            and starts.nplike is reduced.backend.nplike
            and shifts.nplike is reduced.backend.nplike
        )
        reduced.backend.maybe_kernel_error(
            reduced._backend[
                "awkward_NumpyArray_reduce_adjust_starts_shifts_64",
                reduced.dtype.type,
                offsets.dtype.type,
                starts.dtype.type,
                shifts.dtype.type,
            ](
                reduced.data,
                reduced.length,
                offsets.data,
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
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.kind == "c":
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmin_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    outlength,
                )
            )
        else:
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmin",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                    starts.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    starts.data,
                    outlength,
                )
            )
        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        apply_positional_corrections(result_array, offsets, starts, shifts)
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
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.kind == "c":
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmax_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    outlength,
                )
            )
        else:
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_argmax",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                    starts.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    starts.data,
                    outlength,
                )
            )
        result_array = ak.contents.NumpyArray(result, backend=array.backend)
        apply_positional_corrections(result_array, offsets, starts, shifts)
        return result_array


class Count(KernelReducer):
    name: Final = "count"
    preferred_dtype: Final = np.int64
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        assert offsets.nplike is array.backend.nplike
        array.backend.maybe_kernel_error(
            array.backend[
                "awkward_reduce_count_64", result.dtype.type, offsets.dtype.type
            ](
                result,
                offsets.data,
                outlength,
            )
        )
        return ak.contents.NumpyArray(result, backend=array.backend)


class CountNonzero(KernelReducer):
    name: Final = "count_nonzero"
    preferred_dtype: Final = np.int64
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))

        result = array.backend.nplike.empty(outlength, dtype=np.int64)
        if np.issubdtype(array.dtype, np.complexfloating):
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_countnonzero_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    outlength,
                )
            )
        else:
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_countnonzero",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result, backend=array.backend)


class Sum(KernelReducer):
    name: Final = "sum"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, dtype: DTypeLike | None = None):
        # Internal-only forced accumulator dtype (NumPy's umr_sum(dtype=...)).
        # ak.mean/ak.var use dtype=float64 so integer/float32 input accumulates
        # in double precision without materialising a promoted copy. Not exposed
        # on ak.sum.
        self._dtype = None if dtype is None else np.dtype(dtype)

    def axis_none_reducer(self) -> AxisNoneSum:
        return AxisNoneSum(self._dtype)

    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")

        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=self._promote_integer_rank(np.bool_),
            )
            if result.dtype in (np.int64, np.uint64):
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum_int64_bool_64",
                        np.int64,
                        array.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        array.data,
                        offsets.data,
                        outlength,
                    )
                )
            elif result.dtype in (np.int32, np.uint32):
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum_int32_bool_64",
                        np.int32,
                        array.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        array.data,
                        offsets.data,
                        outlength,
                    )
                )
            else:
                raise NotImplementedError
            # bool sums can't overflow; if a float accumulator was requested,
            # cast the small (outlength) result.
            if self._dtype is not None:
                result = array.backend.nplike.astype(result, self._dtype)
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            is_complex = array.dtype.kind == "c"
            # A forced float accumulator (self._dtype) sums directly into that
            # dtype via awkward_reduce_sum_<out>_<in>_64 -- no promoted input
            # copy. Not applied to complex, datetime, or timedelta (kinds c/M/m),
            # which must keep their own dtype (e.g. mean of timedelta must stay
            # timedelta, not become float64 tick counts). Mirrors the jax guard.
            use_forced = self._dtype is not None and array.dtype.kind not in "cmM"
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result_dtype = (
                np.result_type(self._dtype, array.dtype)
                if use_forced and array.dtype.kind == "f"
                else self._dtype
                if use_forced
                else self._promote_integer_rank(kernel_array_data.dtype)
            )
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=result_dtype,
            )
            if is_complex:
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
                        outlength,
                    )
                )
            else:
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_sum",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
                        outlength,
                    )
                )

            if use_forced:
                return ak.contents.NumpyArray(result, backend=array.backend)
            return ak.contents.NumpyArray(
                result.view(self._promote_integer_rank(array.dtype)),
                backend=array.backend,
            )


class AxisNoneSum(Sum):
    def apply(
        self,
        array: ak.contents.NumpyArray,
        _offsets: ak.index.Index,
        _starts: ak.index.Index,
        _shifts: ak.index.Index | None,
        _outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")

        nplike = array.backend.nplike
        reduce_fn = getattr(nplike, self.name)
        if self._dtype is not None and array.dtype.kind != "c":
            dtype = (
                np.result_type(self._dtype, array.dtype)
                if array.dtype.kind == "f"
                else self._dtype
            )
            result_scalar = reduce_fn(array.data, axis=None, dtype=dtype)
        else:
            result_scalar = reduce_fn(array.data, axis=None)
        result_array = nplike.reshape(nplike.asarray(result_scalar), (1,))

        return ak.contents.NumpyArray(result_array, backend=array.backend)


def _statistics_dtype(dtype):
    dtype = np.dtype(dtype)
    return dtype if dtype.kind == "c" else np.result_type(dtype, np.float64)


class SumOfSquares(KernelReducer):
    """Fused sums of squares in a precision-preserving accumulator.

    Real inputs accumulate in at least float64; extended real and complex
    inputs keep their precision. Complex squares are algebraic squares,
    as required by moments, rather than squared magnitudes.
    """

    name: Final = "sumofsquares"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def axis_none_reducer(self) -> AxisNoneSumOfSquares:
        # For a full reduction on a concrete backend, route through NumPy/CuPy's
        # optimized (SIMD/BLAS-class) reduction instead of the scalar per-element
        # kernel loop -- a large speedup for big arrays.
        return AxisNoneSumOfSquares()

    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum-of-squares of {array.dtype!r}")
        dtype = _statistics_dtype(array.dtype)
        is_complex = dtype.kind == "c"
        component_dtype = self._dtype_for_kernel(dtype)
        data = array.data.view(self._dtype_for_kernel(array.dtype))
        length = self._length_for_kernel(dtype, outlength)
        result = array.backend.nplike.empty(length, dtype=component_dtype)

        assert offsets.nplike is array.backend.nplike
        array.backend.maybe_kernel_error(
            array.backend[
                "awkward_reduce_sumofsquares" + ("_complex" if is_complex else ""),
                component_dtype.type,
                data.dtype.type,
                offsets.dtype.type,
            ](
                result,
                data,
                offsets.data,
                outlength,
            )
        )
        return ak.contents.NumpyArray(result.view(dtype), backend=array.backend)


class AxisNoneSumOfSquares(SumOfSquares):
    """``axis=None`` specialization of :class:`SumOfSquares`.

    Chooses the same precision-preserving dtype as the segmented kernel,
    squares with the backend's ufunc, and reduces via ``nplike.sum`` (NumPy/CuPy's optimized reduction) rather than the scalar
    kernel loop. The whole computation stays on the backend's device; only the
    final scalar returns to the host, as for every ``axis=None`` reduction.
    """

    def apply(
        self,
        array: ak.contents.NumpyArray,
        _offsets: ak.index.Index,
        _starts: ak.index.Index,
        _shifts: ak.index.Index | None,
        _outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum-of-squares of {array.dtype!r}")
        nplike = array.backend.nplike
        data = array.data
        dtype = _statistics_dtype(data.dtype)
        widened = nplike.astype(data, dtype)
        result_scalar = nplike.sum(widened * widened, axis=None, dtype=dtype)
        result_array = nplike.reshape(nplike.asarray(result_scalar), (1,))
        return ak.contents.NumpyArray(result_array, backend=array.backend)


class CenteredSumOfSquares(KernelReducer):
    """Fused sums of squared deviations about aligned per-bin means.

    Complex deviations contribute squared magnitudes to a real result.
    The accumulator preserves the precision of both inputs and means.
    """

    name: Final = "sumofsquares"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, means):
        # One mean per output bin, in the reducer descent's bin order.
        self._means = means

    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum-of-squares of {array.dtype!r}")
        dtype = _statistics_dtype(np.result_type(array.dtype, self._means.dtype))
        is_complex = dtype.kind == "c"
        component_dtype = self._dtype_for_kernel(dtype)
        values = array.data
        if (
            is_complex or dtype.type is np.longdouble
        ) and values.dtype.type is not dtype.type:
            values = array.backend.nplike.astype(values, dtype)
        data = values.view(self._dtype_for_kernel(values.dtype))
        length = outlength
        result = array.backend.nplike.empty(length, dtype=component_dtype)
        means = array.backend.nplike.astype(self._means, dtype).view(component_dtype)

        assert offsets.nplike is array.backend.nplike
        array.backend.maybe_kernel_error(
            array.backend[
                "awkward_reduce_centered_sumofsquares"
                + ("_complex" if is_complex else ""),
                component_dtype.type,  # toptr
                data.dtype.type,  # fromptr
                offsets.dtype.type,  # offsets
                component_dtype.type,  # means
            ](
                result,
                data,
                offsets.data,
                outlength,
                means,
            )
        )
        return ak.contents.NumpyArray(result, backend=array.backend)


class SumOfPowers(KernelReducer):
    """Fused sums of integer powers, preserving real or complex precision.

    Real inputs use at least float64; complex inputs retain their component
    precision. The power is computed before accumulation without a full
    intermediate array.
    """

    name: Final = "sumofpowers"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, n: int):
        self._n = int(n)

    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ValueError(f"cannot compute the sum-of-powers of {array.dtype!r}")
        dtype = _statistics_dtype(array.dtype)
        is_complex = dtype.kind == "c"
        component_dtype = self._dtype_for_kernel(dtype)
        data = array.data.view(self._dtype_for_kernel(array.dtype))
        length = self._length_for_kernel(dtype, outlength)
        result = array.backend.nplike.empty(length, dtype=component_dtype)

        assert offsets.nplike is array.backend.nplike
        array.backend.maybe_kernel_error(
            array.backend[
                "awkward_reduce_sumofpowers" + ("_complex" if is_complex else ""),
                component_dtype.type,
                data.dtype.type,
                offsets.dtype.type,
            ](
                result,
                data,
                offsets.data,
                outlength,
                self._n,
            )
        )
        return ak.contents.NumpyArray(result.view(dtype), backend=array.backend)


class Prod(KernelReducer):
    name: Final = "prod"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def apply(
        self,
        array: ak.contents.NumpyArray,
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind.upper() == "M":
            raise ValueError(f"cannot compute the product (ak.prod) of {array.dtype!r}")

        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(
                outlength,
                dtype=np.bool_,
            )
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    array.data,
                    offsets.data,
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
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=self._promote_integer_rank(kernel_array_data.dtype),
            )
            if array.dtype.kind == "c":
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_prod_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
                        outlength,
                    )
                )
            else:
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_prod",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
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
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)

        if array.dtype.kind == "c":
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_sum_bool_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    outlength,
                )
            )
        else:
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_sum_bool",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
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
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
        result = array.backend.nplike.empty(outlength, dtype=np.bool_)

        if array.dtype.kind == "c":
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool_complex",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
                    outlength,
                )
            )
        else:
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    kernel_array_data.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    kernel_array_data,
                    offsets.data,
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

    def axis_none_reducer(self) -> AxisNoneMin:
        return AxisNoneMin(self._initial)

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
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(outlength, dtype=np.bool_)
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    array.data,
                    offsets.data,
                    outlength,
                )
            )
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=kernel_array_data.dtype,
            )
            if array.dtype.kind == "c":
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_min_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            else:
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_min",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )


class AxisNoneMin(Min):
    def apply(
        self,
        array: ak.contents.NumpyArray,
        _offsets: ak.index.Index,
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
            result_scalar = nplike.min(data, axis=None)
            if self.initial is not None:
                initial_val = nplike.asarray(self.initial, dtype=data.dtype)
                result_scalar = nplike.minimum(result_scalar, initial_val)

        result_array = nplike.reshape(nplike.asarray(result_scalar), (1,))

        return ak.contents.NumpyArray(result_array, backend=array.backend)


class Max(KernelReducer):
    name: Final = "max"
    preferred_dtype: Final = np.float64
    needs_position: Final = False

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    def axis_none_reducer(self) -> AxisNoneMax:
        return AxisNoneMax(self._initial)

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
        offsets: ak.index.Index,
        starts: ak.index.Index,
        shifts: ak.index.Index | None,
        outlength: ShapeItem,
    ) -> ak.contents.NumpyArray:
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype == np.bool_:
            result = array.backend.nplike.empty(outlength, dtype=np.bool_)
            assert offsets.nplike is array.backend.nplike
            array.backend.maybe_kernel_error(
                array.backend[
                    "awkward_reduce_sum_bool",
                    result.dtype.type,
                    array.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    array.data,
                    offsets.data,
                    outlength,
                )
            )
            return ak.contents.NumpyArray(result, backend=array.backend)
        else:
            kernel_array_data = array.data.view(self._dtype_for_kernel(array.dtype))
            result = array.backend.nplike.empty(
                self._length_for_kernel(array.dtype.type, outlength),
                dtype=kernel_array_data.dtype,
            )
            if array.dtype.kind == "c":
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_max_complex",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            else:
                assert offsets.nplike is array.backend.nplike
                array.backend.maybe_kernel_error(
                    array.backend[
                        "awkward_reduce_max",
                        result.dtype.type,
                        kernel_array_data.dtype.type,
                        offsets.dtype.type,
                    ](
                        result,
                        kernel_array_data,
                        offsets.data,
                        outlength,
                        self._identity_for(result.dtype),
                    )
                )
            return ak.contents.NumpyArray(
                result.view(array.dtype), backend=array.backend
            )


class AxisNoneMax(Max):
    def apply(
        self,
        array: ak.contents.NumpyArray,
        _offsets: ak.index.Index,
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
            result_scalar = nplike.max(data, axis=None)
            if self.initial is not None:
                initial_val = nplike.asarray(self.initial, dtype=data.dtype)
                result_scalar = nplike.maximum(result_scalar, initial_val)

        result_array = nplike.reshape(nplike.asarray(result_scalar), (1,))

        return ak.contents.NumpyArray(result_array, backend=array.backend)
