# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
"""
Most of the nplikes can re-use implementation shared between CuPy, NumPy, and JAX. This module
defines a robust interface to these numpy-like libraries.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy

from awkward import _errors
from awkward._nplikes import Array, NumpyLike, dtypes
from awkward.typing import Any, Literal, Self, SupportsInt


class ArrayModuleArray(Array):

    __slots__ = ("_array", "_nplike")

    _array: Any
    _nplike: ArrayModuleNumpyLike

    @property
    def dtype(self) -> dtypes.dtype:
        return self._array.dtype  # type: ignore

    @property
    def ndim(self) -> int:
        return self._array.ndim  # type: ignore

    @property
    def shape(self) -> tuple[SupportsInt, ...]:
        return self._array.shape  # type: ignore

    @property
    def size(self) -> SupportsInt:
        return self._array.size  # type: ignore

    @property
    def T(self) -> Self:
        return self._new(self._array.T, nplike=self._nplike)  # type: ignore

    def __new__(cls, *args, **kwargs):
        raise _errors.wrap_error(
            TypeError(
                "internal_error: the `Numpy` nplike's `Array` object should never be directly instantiated"
            )
        )

    @classmethod
    def _new(cls, x, nplike: ArrayModuleNumpyLike) -> Self:
        obj = super().__new__(cls)
        if isinstance(x, numpy.generic):
            x = nplike.array_module.asarray(x)
        if not isinstance(x, nplike.array_module.ndarray):
            raise _errors.wrap_error(
                TypeError(
                    "internal_error: the `Numpy` nplike's `Array` object be created from non-scalars",
                    type(x),
                )
            )
        obj._array = x  # type: ignore
        obj._nplike = nplike
        return obj

    def __add__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__add__(right._array), nplike=self._nplike)  # type: ignore

    def __sub__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__sub__(right._array), nplike=self._nplike)  # type: ignore

    def __truediv__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__truediv__(right._array), nplike=self._nplike)  # type: ignore

    def __floordiv__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__floordiv__(right._array), nplike=self._nplike)  # type: ignore

    def __mod__(self, other: int | float | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__mod__(right._array), nplike=self._nplike)  # type: ignore

    def __mul__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__mul__(right._array), nplike=self._nplike)  # type: ignore

    def __pow__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__pow__(right._array), nplike=self._nplike)  # type: ignore

    def __xor__(self, other: int | bool | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__xor__(right._array), nplike=self._nplike)  # type: ignore

    def __and__(self, other: int | bool | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__and__(right._array), nplike=self._nplike)  # type: ignore

    def __or__(self, other: int | bool | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__or__(right._array), nplike=self._nplike)  # type: ignore

    def __lt__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__lt__(right._array), nplike=self._nplike)  # type: ignore

    def __le__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__lt__(right._array), nplike=self._nplike)  # type: ignore

    def __gt__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__gt__(right._array), nplike=self._nplike)  # type: ignore

    def __ge__(self, other: int | float | complex | Self) -> Self:
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__ge__(right._array), nplike=self._nplike)  # type: ignore

    def __eq__(self, other: int | float | bool | complex | Self) -> Self:  # type: ignore
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__eq__(right._array), nplike=self._nplike)

    def __ne__(self, other: int | float | bool | complex | Self) -> Self:  # type: ignore
        other = self._nplike.promote_scalar(other)
        left, right = self._nplike.normalise_binary_arguments(self, other)
        return left._new(left._array.__ne__(right._array), nplike=self._nplike)  # type: ignore

    def __abs__(self) -> Self:
        return self._new(self._array.__abs__(), nplike=self._nplike)  # type: ignore

    def __neg__(self) -> Self:
        return self._new(self._array.__neg__(), nplike=self._nplike)  # type: ignore

    def __pos__(self) -> Self:
        return self._new(self._array.__pos__(), nplike=self._nplike)  # type: ignore

    def __invert__(self) -> Self:
        return self._new(self._array.__invert__(), nplike=self._nplike)  # type: ignore

    def __bool__(self) -> bool:
        return self._array.__bool__()  # type: ignore

    def __int__(self) -> int:
        return self._array.__int__()  # type: ignore

    def __index__(self) -> int:
        return self._array.__index__()  # type: ignore

    def __str__(self) -> str:
        return str(self._array)

    def __repr__(self) -> str:
        return f"<{self._nplike.__class__.__name__} :: {self._array!r}>"


class ArrayModuleNumpyLike(NumpyLike[ArrayModuleArray]):
    """
    An abstract class implementing NumpyLike support for a `numpy_api` module e.g. numpy, cupy
    """

    known_data: bool
    known_shape: bool
    is_eager: bool

    def promote_scalar(
        self, x: bool | int | float | complex | ArrayModuleArray
    ) -> ArrayModuleArray:
        if isinstance(x, float):
            return ArrayModuleArray._new(
                self.array_module.array(x, dtype=numpy.float64), nplike=self
            )
        elif isinstance(x, bool):
            return ArrayModuleArray._new(
                self.array_module.array(x, dtype=numpy.bool_), nplike=self
            )
        elif isinstance(x, int):
            return ArrayModuleArray._new(
                self.array_module.array(x, dtype=numpy.int64), nplike=self
            )
        elif isinstance(x, complex):
            return ArrayModuleArray._new(
                self.array_module.array(x, dtype=numpy.complex128), nplike=self
            )
        elif not isinstance(x, ArrayModuleArray):
            raise _errors.wrap_error(TypeError("Expected bool, int, float, or Array"))
        else:
            return x

    def normalise_binary_arguments(
        self,
        left: ArrayModuleArray,
        right: ArrayModuleArray,
    ) -> tuple[ArrayModuleArray, ArrayModuleArray]:
        if left.ndim == 0 and right.ndim != 0:
            left = ArrayModuleArray._new(left._array[numpy.newaxis], nplike=self)
        elif right.ndim != 0 and right.ndim == 0:
            right = ArrayModuleArray._new(right._array[numpy.newaxis], nplike=self)
        return left, right

    ############################ array creation

    @property
    @abstractmethod
    def array_module(self):
        raise _errors.wrap_error(NotImplementedError)

    def asarray(
        self,
        obj,
        *,
        dtype: dtypes.dtype | None = None,
        copy: bool | None = None,
    ) -> ArrayModuleArray:
        # Unwrap Array object
        if isinstance(obj, ArrayModuleArray):
            obj = obj._array
        if copy:
            array = self.array_module.array(obj, dtype=dtype, copy=True)
        elif copy is None:
            array = self.array_module.asarray(obj, dtype=dtype)
        else:
            assert not copy
            raise _errors.wrap_error(
                ValueError("internal error: copy=False is not supported yet")
            )
        return ArrayModuleArray._new(array, nplike=self)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.zeros(shape, dtype=dtype), nplike=self
        )

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.ones(shape, dtype=dtype), nplike=self
        )

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtypes.dtype | None = None,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.empty(shape, dtype=dtype), nplike=self
        )

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: dtypes.dtype | None = None,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.full(shape, dtype=dtype), nplike=self
        )

    def zeros_like(
        self, x: ArrayModuleArray, *, dtype: dtypes.dtype | None = None
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.zeros_like(x._array, dtype=dtype), nplike=self
        )

    def ones_like(
        self, x: ArrayModuleArray, *, dtype: dtypes.dtype | None = None
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.ones_like(x._array, dtype=dtype), nplike=self
        )

    def full_like(
        self, x: ArrayModuleArray, fill_value, *, dtype: dtypes.dtype | None = None
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.full_like(x._array, dtype=dtype), nplike=self
        )

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: dtypes.dtype | None = None,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.arange(start, stop, step, dtype=dtype), nplike=self
        )

    def meshgrid(
        self, *arrays: ArrayModuleArray, indexing: Literal["xy", "ij"] = "xy"
    ) -> ArrayModuleArray:
        raw_arrays = [x._array for x in arrays]
        return ArrayModuleArray._new(
            self.array_module.meshgrid(*raw_arrays, indexing=indexing), nplike=self
        )

    ############################ data type functions

    def astype(
        self, x: ArrayModuleArray, dtype: dtypes.dtype, *, copy: bool = True
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            x._array.astype(dtype=dtype, copy=copy), nplike=self
        )

    def broadcast_arrays(self, *arrays: ArrayModuleArray) -> list[ArrayModuleArray]:
        raw_arrays = [x._array for x in arrays]
        return [
            ArrayModuleArray._new(x, nplike=self)
            for x in self.array_module.broadcast_arrays(*raw_arrays)
        ]

    def broadcast_to(
        self, x: ArrayModuleArray, shape: tuple[SupportsInt, ...]
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.broadcast_to(x._array, shape), nplike=self
        )

    def result_type(
        self, *arrays_and_dtypes: ArrayModuleArray | dtypes.dtype
    ) -> dtypes.dtype:
        all_dtypes: list[dtypes.dtype] = []
        for item in arrays_and_dtypes:
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                item = item.dtype
            if isinstance(item, dtypes.dtype):
                all_dtypes.append(item)
            else:
                raise _errors.wrap_error(
                    TypeError("result_type() inputs must be array_api arrays or dtypes")
                )

        return numpy.result_type(*all_dtypes)

    ############################ searching functions

    def nonzero(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(self.array_module.nonzero(x._array), nplike=self)

    def where(
        self, condition: ArrayModuleArray, x1: ArrayModuleArray, x2: ArrayModuleArray
    ) -> ArrayModuleArray:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return ArrayModuleArray._new(
            self.array_module.where(condition._array, x1._array, x2._array), nplike=self
        )

    ############################ set functions

    def unique_counts(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.unique(
                x._array,
                return_counts=True,
                return_index=False,
                return_inverse=False,
                equal_nan=False,
            ),
            nplike=self,
        )

    def unique_values(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.unique(
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
        arrays: list[ArrayModuleArray] | tuple[ArrayModuleArray, ...],
        *,
        axis: int | None = 0,
    ) -> ArrayModuleArray:
        arrays = [x._array for x in arrays]
        return ArrayModuleArray._new(
            self.array_module.concatenate(arrays, axis=axis), nplike=self
        )

    def stack(
        self,
        arrays: list[ArrayModuleArray] | tuple[ArrayModuleArray, ...],
        *,
        axis: int = 0,
    ) -> ArrayModuleArray:
        arrays = [x._array for x in arrays]
        return ArrayModuleArray._new(
            self.array_module.stack(arrays, axis=axis), nplike=self
        )

    ############################ ufuncs

    def add(self, x1: ArrayModuleArray, x2: ArrayModuleArray) -> ArrayModuleArray:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return ArrayModuleArray._new(
            self.array_module.add(x1._array, x2._array), nplike=self
        )

    def multiply(self, x1: ArrayModuleArray, x2: ArrayModuleArray) -> ArrayModuleArray:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return ArrayModuleArray._new(
            self.array_module.multiply(x1._array, x2._array), nplike=self
        )

    def logical_or(
        self, x1: ArrayModuleArray, x2: ArrayModuleArray
    ) -> ArrayModuleArray:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return ArrayModuleArray._new(
            self.array_module.logical_or(x1._array, x2._array), nplike=self
        )

    def logical_and(
        self, x1: ArrayModuleArray, x2: ArrayModuleArray
    ) -> ArrayModuleArray:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return ArrayModuleArray._new(
            self.array_module.logical_and(x1._array, x2._array), nplike=self
        )

    def logical_not(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.logical_not(x._array), nplike=self
        )

    def sqrt(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(self.array_module.sqrt(x), nplike=self)

    def exp(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(self.array_module.exp(x._array), nplike=self)

    def divide(self, x1: ArrayModuleArray, x2: ArrayModuleArray) -> ArrayModuleArray:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return ArrayModuleArray._new(
            self.array_module.divide(x1._array, x2._array), nplike=self
        )

    def equal(self, x1: ArrayModuleArray, x2: ArrayModuleArray) -> ArrayModuleArray:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return ArrayModuleArray._new(
            self.array_module.equal(x1._array, x2._array), nplike=self
        )

    def isnan(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(self.array_module.isnan(x._array), nplike=self)

    ############################ Utility functions

    def all(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.all(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    def any(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.any(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    ############################ Statistical functions

    def sum(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtypes.dtype | None = None,
        keepdims: bool = False,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.sum(x._array, axis=axis, dtype=dtype, keepdims=keepdims),
            nplike=self,
        )

    def prod(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtypes.dtype | None = None,
        keepdims: bool = False,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.prod(x._array, axis=axis, dtype=dtype, keepdims=keepdims),
            nplike=self,
        )

    def min(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.min(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    def max(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.max(x._array, axis=axis, keepdims=keepdims), nplike=self
        )

    ############################ Searching functions

    def argmin(
        self, x: ArrayModuleArray, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.argmin(x._array, axis=axis, keepdims=keepdims),
            nplike=self,
        )

    def argmax(
        self, x: ArrayModuleArray, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.argmin(x._array, axis=axis, keepdims=keepdims),
            nplike=self,
        )

    ############################ extensions to Array API

    def as_contiguous(self, x: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.ascontiguousarray(x._array), nplike=self
        )

    def cumsum(
        self, x: ArrayModuleArray, *, axis: int | None = None
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.cumsum(x._array, axis=axis), nplike=self
        )

    def from_buffer(self, buffer, *, dtype=None) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.frombuffer(buffer, dtype=dtype), nplike=self
        )

    def array_equal(
        self, x1: ArrayModuleArray, x2: ArrayModuleArray, *, equal_nan: bool = False
    ) -> bool:
        return self.array_module.array_equal(x1._array, x2._array, equal_nan=equal_nan)

    def search_sorted(
        self,
        x: ArrayModuleArray,
        values: ArrayModuleArray,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.searchsorted(
                x._array, values._array, side=side, sorter=sorter
            ),
            nplike=self,
        )

    def repeat(
        self,
        x: ArrayModuleArray,
        repeats: ArrayModuleArray | int,
        *,
        axis: int | None = None,
    ) -> ArrayModuleArray:
        if isinstance(repeats, ArrayModuleArray):
            repeats = repeats._array

        return ArrayModuleArray._new(
            self.array_module.repeat(x._array, repeats, axis=axis), nplike=self
        )

    def tile(self, x: ArrayModuleArray, reps: int) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.tile(x._array, reps), nplike=self
        )

    def pack_bits(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.packbits(x._array, axis=axis, bitorder=bitorder),
            nplike=self,
        )

    def unpack_bits(
        self,
        x: ArrayModuleArray,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.unpackbits(
                x._array, axis=axis, count=count, bitorder=bitorder
            ),
            nplike=self,
        )

    def minimum(self, x1: ArrayModuleArray, x2: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.minimum(x1._array, x2._array), nplike=self
        )

    def maximum(self, x1: ArrayModuleArray, x2: ArrayModuleArray) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.maximum(x1._array, x2._array), nplike=self
        )

    def nan_to_num(
        self,
        x: ArrayModuleArray,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.nan_to_num(
                x._array, copy=copy, nan=nan, posinf=posinf, neginf=neginf
            ),
            nplike=self,
        )

    def is_close(
        self,
        x1: ArrayModuleArray,
        x2: ArrayModuleArray,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.isclose(
                x1._array, x2._array, rtol=rtol, atol=atol, equal_nan=equal_nan
            ),
            nplike=self,
        )

    def count_nonzero(
        self, x: ArrayModuleArray, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayModuleArray:
        return ArrayModuleArray._new(
            self.array_module.count_nonzero(x._array, axis=axis, keepdims=keepdims),
            nplike=self,
        )

    def array_str(
        self,
        x: ArrayModuleArray,
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

    def is_c_contiguous(self, x: ArrayModuleArray) -> bool:
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
