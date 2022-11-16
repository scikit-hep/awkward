# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypeVar

import numpy

from awkward import _errors
from awkward.nplikes import dtypes, numpylike

# How will CuPy work? Same Array object? what about typetracer?

ShapeItem = int
Shape = tuple[ShapeItem]
RawArray = TypeVar("RawArray")


class NumpyModuleArray(numpylike.Array):
    _array: RawArray
    _nplike: NumpyModuleLike

    @property
    def dtype(self) -> dtypes.dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def shape(self) -> Shape:
        return self._array.shape

    @property
    def size(self) -> ShapeItem:
        return self._array.size

    @property
    def T(self) -> NumpyModuleArray:
        return self._new(self._array.T, nplike=self._nplike)

    def __new__(cls, *args, **kwargs):
        raise _errors.wrap_error(
            TypeError(
                "internal_error: the `Numpy` nplike's `Array` object should never be directly instantiated"
            )
        )

    @classmethod
    def _new(cls, x, nplike: NumpyModuleLike) -> NumpyModuleArray:
        obj = super().__new__(cls)
        if isinstance(x, numpy.generic):
            x = nplike.numpy_api.asarray(x)
        if not isinstance(x, numpy.ndarray):
            raise _errors.wrap_error(
                TypeError(
                    "internal_error: the `Numpy` nplike's `Array` object be created from non-scalars",
                    type(x),
                )
            )
        obj._array = x
        obj._nplike = nplike
        return obj

    def __add__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__add__(right._array), nplike=self._nplike)

    def __sub__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__sub__(right._array), nplike=self._nplike)

    def __truediv__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__truediv__(right._array), nplike=self._nplike)

    def __floordiv__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__floordiv__(right._array), nplike=self._nplike)

    def __mod__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__mod__(right._array), nplike=self._nplike)

    def __mul__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__mul__(right._array), nplike=self._nplike)

    def __pow__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__pow__(right._array), nplike=self._nplike)

    def __xor__(self, other: int | bool | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__xor__(right._array), nplike=self._nplike)

    def __and__(self, other: int | bool | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__and__(right._array), nplike=self._nplike)

    def __or__(self, other: int | bool | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__or__(right._array), nplike=self._nplike)

    def __lt__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__lt__(right._array), nplike=self._nplike)

    def __le__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__lt__(right._array), nplike=self._nplike)

    def __gt__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__gt__(right._array), nplike=self._nplike)

    def __ge__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__ge__(right._array), nplike=self._nplike)

    def __eq__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__eq__(right._array), nplike=self._nplike)

    def __ne__(self, other: int | float | NumpyModuleArray) -> NumpyModuleArray:
        other = self._nplike._promote_scalar(other)
        left, right = self._nplike._normalise_binary_arguments(self, other)
        return left._new(left._array.__ne__(right._array), nplike=self._nplike)

    def __abs__(self) -> NumpyModuleArray:
        return self._new(self._array.__abs__(), nplike=self._nplike)

    def __neg__(self) -> NumpyModuleArray:
        return self._new(self._array.__neg__(), nplike=self._nplike)

    def __pos__(self) -> NumpyModuleArray:
        return self._new(self._array.__pos__(), nplike=self._nplike)

    def __invert__(self) -> NumpyModuleArray:
        return self._new(self._array.__invert__(), nplike=self._nplike)

    def __bool__(self) -> bool:
        return self._array.__bool__()

    def __int__(self) -> int:
        return self._array.__int__()

    def __index__(self) -> int:
        return self._array.__index__()


class NumpyModuleLike(numpylike.NumpyLike[NumpyModuleArray]):
    """
    An abstract class implementing NumpyLike support for a `numpy_api` module e.g. numpy, cupy
    """

    known_data: bool
    known_shape: bool
    is_eager: bool

    ############################ array creation

    @property
    @abstractmethod
    def numpy_api(self):
        raise NotImplementedError

    def _promote_scalar(
        self, x: bool | int | float | complex | NumpyModuleArray
    ) -> NumpyModuleArray:
        if isinstance(x, float):
            return NumpyModuleArray._new(
                self.numpy_api.array(x, dtype=numpy.float64), self
            )
        elif isinstance(x, bool):
            return NumpyModuleArray._new(
                self.numpy_api.array(x, dtype=numpy.bool_), self
            )
        elif isinstance(x, int):
            return NumpyModuleArray._new(
                self.numpy_api.array(x, dtype=numpy.int64), self
            )
        elif isinstance(x, complex):
            return NumpyModuleArray._new(
                self.numpy_api.array(x, dtype=numpy.complex128), self
            )
        elif not isinstance(x, NumpyModuleArray):
            raise _errors.wrap_error(TypeError("Expected bool, int, float, or Array"))
        else:
            return x

    def _normalise_binary_arguments(
        self,
        left: NumpyModuleArray,
        right: NumpyModuleArray,
    ) -> tuple[NumpyModuleArray, NumpyModuleArray]:
        if left.ndim == 0 and right.ndim != 0:
            left = NumpyModuleArray._new(left._array[numpy.newaxis], self)
        elif right.ndim != 0 and right.ndim == 0:
            right = NumpyModuleArray._new(right._array[numpy.newaxis], self)
        return left, right

    def asarray(
        self,
        obj,
        *,
        dtype: dtypes.dtype | None = None,
        copy: bool | None = None,
    ) -> NumpyModuleArray:
        # Unwrap Array object
        if isinstance(obj, NumpyModuleArray):
            obj = obj._array
        if copy:
            array = self.numpy_api.array(obj, dtype=dtype, copy=True)
        elif copy is None:
            array = self.numpy_api.asarray(obj, dtype=dtype)
        else:
            assert not copy
            raise _errors.wrap_error(
                ValueError("internal error: copy=False is not supported yet")
            )
        return NumpyModuleArray._new(array, nplike=self)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.zeros(shape, dtype=dtype), nplike=self
        )

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.ones(shape, dtype=dtype), nplike=self
        )

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.empty(shape, dtype=dtype), nplike=self
        )

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: dtypes.dtype | None = None,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.full(shape, dtype=dtype), nplike=self
        )

    def zeros_like(
        self, x: NumpyModuleArray, *, dtype: dtypes.dtype | None = None
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.zeros_like(x._array, dtype=dtype), nplike=self
        )

    def ones_like(
        self, x: NumpyModuleArray, *, dtype: dtypes.dtype | None = None
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.ones_like(x._array, dtype=dtype), nplike=self
        )

    def full_like(
        self, x: NumpyModuleArray, fill_value, *, dtype: dtypes.dtype | None = None
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.full_like(x._array, dtype=dtype), nplike=self
        )

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: dtypes.dtype | None = None,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.arange(start, stop, step, dtype=dtype), nplike=self
        )

    def meshgrid(
        self, *arrays: NumpyModuleArray, indexing: Literal["xy", "ij"] = "xy"
    ) -> NumpyModuleArray:
        arrays = [x._array for x in arrays]
        return NumpyModuleArray._new(
            self.numpy_api.meshgrid(*arrays, indexing=indexing), nplike=self
        )

    ############################ data type functions

    def astype(
        self, x: NumpyModuleArray, dtype: dtypes.dtype, *, copy: bool = True
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            x._array.astype(dtype=dtype, copy=copy), nplike=self
        )

    def broadcast_arrays(self, *arrays: NumpyModuleArray) -> list[NumpyModuleArray]:
        arrays = [x._array for x in arrays]
        return [
            NumpyModuleArray._new(x, nplike=self)
            for x in self.numpy_api.broadcast_arrays(*arrays)
        ]

    def broadcast_to(self, x: NumpyModuleArray, shape: Shape) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.broadcast_to(x._array, shape), nplike=self
        )

    def result_type(self, *arrays_and_dtypes: NumpyModuleArray) -> dtypes.dtype:
        all_dtypes: list[dtypes.dtype] = []
        for item in arrays_and_dtypes:
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                item = item.dtype
            if isinstance(item, dtypes.dtype):
                all_dtypes.append(item)
            else:
                raise TypeError(
                    "result_type() inputs must be array_api arrays or dtypes"
                )

        return numpy.result_type(*all_dtypes)

    ############################ searching functions

    def nonzero(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(self.numpy_api.nonzero(x._array), nplike=self)

    def where(
        self, condition: NumpyModuleArray, x1: NumpyModuleArray, x2: NumpyModuleArray
    ) -> NumpyModuleArray:
        x1, x2 = self._normalise_binary_arguments(x1, x2)
        return NumpyModuleArray._new(
            self.numpy_api.where(condition._array, x1._array, x2._array), nplike=self
        )

    ############################ set functions

    def unique_counts(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.unique(
                x._array,
                return_counts=True,
                return_index=False,
                return_inverse=False,
                equal_nan=False,
            ),
            nplike=self,
        )

    def unique_values(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.unique(
                x._array,
                return_counts=False,
                return_index=False,
                return_inverse=False,
                equal_nan=False,
            ),
            nplike=self,
        )

    ############################ manipulation functions

    def concat(
        self,
        arrays: list[NumpyModuleArray] | tuple[NumpyModuleArray, ...],
        *,
        axis: int | None = 0,
    ) -> NumpyModuleArray:
        arrays = [x._array for x in arrays]
        return NumpyModuleArray._new(
            self.numpy_api.concatenate(arrays, axis=axis), nplike=self
        )

    def stack(
        self,
        arrays: list[NumpyModuleArray] | tuple[NumpyModuleArray, ...],
        *,
        axis: int = 0,
    ) -> NumpyModuleArray:
        arrays = [x._array for x in arrays]
        return NumpyModuleArray._new(
            self.numpy_api.stack(arrays, axis=axis), nplike=self
        )

    ############################ ufuncs

    def add(self, x1: NumpyModuleArray, x2: NumpyModuleArray) -> NumpyModuleArray:
        x1, x2 = self._normalise_binary_arguments(x1, x2)
        return NumpyModuleArray._new(
            self.numpy_api.add(x1._array, x2._array), nplike=self
        )

    def multiply(self, x1: NumpyModuleArray, x2: NumpyModuleArray) -> NumpyModuleArray:
        x1, x2 = self._normalise_binary_arguments(x1, x2)
        return NumpyModuleArray._new(
            self.numpy_api.multiply(x1._array, x2._array), nplike=self
        )

    def logical_or(
        self, x1: NumpyModuleArray, x2: NumpyModuleArray
    ) -> NumpyModuleArray:
        x1, x2 = self._normalise_binary_arguments(x1, x2)
        return NumpyModuleArray._new(
            self.numpy_api.logical_or(x1._array, x2._array), nplike=self
        )

    def logical_and(
        self, x1: NumpyModuleArray, x2: NumpyModuleArray
    ) -> NumpyModuleArray:
        x1, x2 = self._normalise_binary_arguments(x1, x2)
        return NumpyModuleArray._new(
            self.numpy_api.logical_and(x1._array, x2._array), nplike=self
        )

    def logical_not(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(self.numpy_api.logical_not(x._array), nplike=self)

    def sqrt(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(self.numpy_api.sqrt(x), nplike=self)

    def exp(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(self.numpy_api.exp(x._array), nplike=self)

    def divide(self, x1: NumpyModuleArray, x2: NumpyModuleArray) -> NumpyModuleArray:
        x1, x2 = self._normalise_binary_arguments(x1, x2)
        return NumpyModuleArray._new(
            self.numpy_api.divide(x1._array, x2._array), nplike=self
        )

    def equal(self, x1: NumpyModuleArray, x2: NumpyModuleArray) -> NumpyModuleArray:
        x1, x2 = self._normalise_binary_arguments(x1, x2)
        return NumpyModuleArray._new(
            self.numpy_api.equal(x1._array, x2._array), nplike=self
        )

    def isnan(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(self.numpy_api.isnan(x._array), nplike=self)

    ############################ Utility functions

    def all(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.all(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    def any(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.any(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    ############################ Statistical functions

    def sum(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtypes.dtype | None = None,
        keepdims: bool = False,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.sum(x._array, axis=axis, dtype=dtype, keepdims=keepdims),
            nplike=self,
        )

    def prod(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtypes.dtype | None = None,
        keepdims: bool = False,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.prod(x._array, axis=axis, dtype=dtype, keepdims=keepdims),
            nplike=self,
        )

    def min(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.min(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    def max(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.max(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    ############################ Searching functions

    def argmin(
        self, x: NumpyModuleArray, *, axis: int | None = None, keepdims: bool = False
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.argmin(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    def argmax(
        self, x: NumpyModuleArray, *, axis: int | None = None, keepdims: bool = False
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.argmin(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    ############################ extensions to Array API

    def ascontiguousarray(self, x: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.ascontiguousarray(x._array), nplike=self
        )

    def cumsum(
        self, x: NumpyModuleArray, *, axis: int | None = None
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.cumsum(x._array, axis=axis), nplike=self
        )

    def from_buffer(self, buffer, *, dtype=None) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.frombuffer(buffer, dtype=dtype), nplike=self
        )

    def array_equal(
        self, x1: NumpyModuleArray, x2: NumpyModuleArray, *, equal_nan: bool = False
    ) -> bool:
        return self.numpy_api.array_equal(x1._array, x2._array, equal_nan=equal_nan)

    def search_sorted(
        self,
        x: NumpyModuleArray,
        values: NumpyModuleArray,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.searchsorted(
                x._array, values._array, side=side, sorter=sorter
            ),
            nplike=self,
        )

    def repeat(
        self,
        x: NumpyModuleArray,
        repeats: NumpyModuleArray | int,
        *,
        axis: int | None = None,
    ) -> NumpyModuleArray:
        if isinstance(repeats, NumpyModuleArray):
            repeats = repeats._array

        return NumpyModuleArray._new(
            self.numpy_api.repeat(x._array, repeats, axis=axis), nplike=self
        )

    def tile(self, x: NumpyModuleArray, reps: int) -> NumpyModuleArray:
        return NumpyModuleArray._new(self.numpy_api.tile(x._array, reps), nplike=self)

    def pack_bits(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big -> Array",
    ):
        return NumpyModuleArray._new(
            self.numpy_api.packbits(x._array, axis=axis, bitorder=bitorder), nplike=self
        )

    def unpack_bits(
        self,
        x: NumpyModuleArray,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.unpackbits(
                x._array, axis=axis, count=count, bitorder=bitorder
            ),
            nplike=self,
        )

    def minimum(self, x1: NumpyModuleArray, x2: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.minimum(x1._array, x2._array), nplike=self
        )

    def maximum(self, x1: NumpyModuleArray, x2: NumpyModuleArray) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.maximum(x1._array, x2._array), nplike=self
        )

    def nan_to_num(
        self,
        x: NumpyModuleArray,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.nan_to_num(
                x._array, copy=copy, nan=nan, posinf=posinf, neginf=neginf
            ),
            nplike=self,
        )

    def is_close(
        self,
        x1: NumpyModuleArray,
        x2: NumpyModuleArray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.isclose(
                x1._array, x2._array, rtol=rtol, atol=atol, equal_nan=equal_nan
            ),
            nplike=self,
        )

    def count_nonzero(
        self, x: NumpyModuleArray, *, axis: int | None = None, keepdims: bool = False
    ) -> NumpyModuleArray:
        return NumpyModuleArray._new(
            self.numpy_api.count_nonzero(x._array, axis=axis, keepdims=keepdims),
            nplike=self,
        )

    def array_str(
        self,
        x: NumpyModuleArray,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        return numpy.array_str(
            x._array,
            max_line_width=max_line_width,
            precision=precision,
            suppress_small=suppress_small,
        )

    def is_c_contiguous(self, x: NumpyModuleArray) -> bool:
        return x._array.flags["C_CONTIGUOUS"]

    def to_rectilinear(self, array):
        raise _errors.wrap_error(NotImplementedError)

    @classmethod
    def is_own_array(cls, x) -> bool:
        """
        Args:
            x: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        raise _errors.wrap_error(NotImplementedError)


class Numpy(NumpyModuleLike):
    """
    A concrete class importing `NumpyModuleLike` for `numpy`
    """

    numpy_api = numpy
