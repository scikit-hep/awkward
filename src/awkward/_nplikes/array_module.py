# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
"""
Most of the nplikes can re-use implementation shared between CuPy, NumPy, and JAX. This module
defines a robust interface to these numpy-like libraries.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Literal, SupportsInt

import numpy

from awkward import _errors
from awkward._nplikes import Array, NumpyLike, metadata
from awkward._nplikes.numpylike import ErrorStateLiteral
from awkward.typing import ContextManager, Final


class ArrayModuleNumpyLike(NumpyLike):
    """
    An abstract class implementing NumpyLike support for a `numpy_api` module e.g. numpy, cupy
    """

    known_data: Final[bool] = True
    known_shape: Final[bool] = True

    def promote_scalar(self, x: bool | int | float | complex | Array) -> Array:
        if isinstance(x, self.array_module.ndarray):
            return x
        elif isinstance(x, float):
            return self.array_module.array(x, dtype=numpy.float64)
        elif isinstance(x, bool):
            return self.array_module.array(x, dtype=numpy.bool_)
        elif isinstance(x, int):
            return self.array_module.array(x, dtype=numpy.int64)
        elif isinstance(x, complex):
            return self.array_module.array(x, dtype=numpy.complex128)
        # Scalars must become 0-D arrays
        elif isinstance(x, numpy.generic):
            return self.array_module.asarray(x)
        else:
            raise _errors.wrap_error(
                TypeError(
                    f"Expected bool, int, float, complex, or {self.array_module.ndarray}"
                )
            )

    def normalise_binary_arguments(
        self,
        left: Array,
        right: Array,
    ) -> tuple[Array, Array]:
        if left.ndim == 0 and right.ndim != 0:
            left = left[numpy.newaxis]
        elif right.ndim != 0 and right.ndim == 0:
            right = right[numpy.newaxis]
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
        dtype: metadata.dtype | None = None,
        copy: bool | None = None,
    ) -> Array:
        # Unwrap Array object
        if copy:
            array = self.array_module.array(obj, dtype=dtype, copy=True)
        elif copy is None:
            array = self.array_module.asarray(obj, dtype=dtype)
        else:
            assert not copy
            raise _errors.wrap_error(
                ValueError("internal error: copy=False is not supported yet")
            )
        return array

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> Array:
        return self.array_module.zeros(shape, dtype=dtype)

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> Array:
        return self.array_module.ones(shape, dtype=dtype)

    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: metadata.dtype | None = None,
    ) -> Array:
        return self.array_module.empty(shape, dtype=dtype)

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: metadata.dtype | None = None,
    ) -> Array:
        return self.array_module.full(shape, fill_value, dtype=dtype)

    def zeros_like(self, x: Array, *, dtype: metadata.dtype | None = None) -> Array:
        return self.array_module.zeros_like(x, dtype=dtype)

    def ones_like(self, x: Array, *, dtype: metadata.dtype | None = None) -> Array:
        return self.array_module.ones_like(x, dtype=dtype)

    def full_like(
        self, x: Array, fill_value, *, dtype: metadata.dtype | None = None
    ) -> Array:
        return self.array_module.full_like(x, fill_value, dtype=dtype)

    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: metadata.dtype | None = None,
    ) -> Array:
        return self.array_module.arange(start, stop, step, dtype=dtype)

    def meshgrid(self, *arrays: Array, indexing: Literal["xy", "ij"] = "xy") -> Array:
        raw_arrays = [x for x in arrays]
        return self.array_module.meshgrid(*raw_arrays, indexing=indexing)

    ############################ data type functions

    def astype(self, x: Array, dtype: metadata.dtype, *, copy: bool = True) -> Array:
        return x.astype(dtype=dtype, copy=copy)  # type: ignore

    def broadcast_arrays(self, *arrays: Array) -> list[Array]:
        raw_arrays = [x for x in arrays]
        return [x for x in self.array_module.broadcast_arrays(*raw_arrays)]

    def broadcast_to(self, x: Array, shape: tuple[SupportsInt, ...]) -> Array:
        return self.array_module.broadcast_to(x, shape)

    ############################ searching functions

    def nonzero(self, x: Array) -> Array:
        return self.array_module.nonzero(x)

    def where(self, condition: Array, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.where(condition, x1, x2)

    ############################ set functions

    def unique_counts(self, x: Array) -> Array:
        return self.array_module.unique(
            x,
            return_counts=True,
            return_index=False,
            return_inverse=False,
            equal_nan=False,
        )

    def unique_values(self, x: Array) -> Array:
        return self.array_module.unique(
            x,
            return_counts=False,
            return_index=False,
            return_inverse=False,
            equal_nan=False,
        )

    ############################ manipulation functions

    def concat(
        self,
        arrays: list[Array] | tuple[Array, ...],
        *,
        axis: int | None = 0,
    ) -> Array:
        arrays = [x for x in arrays]
        return self.array_module.concatenate(arrays, axis=axis)

    def stack(
        self,
        arrays: list[Array] | tuple[Array, ...],
        *,
        axis: int = 0,
    ) -> Array:
        arrays = [x for x in arrays]
        return self.array_module.stack(arrays, axis=axis)

    ############################ ufuncs

    def add(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.add(x1, x2)

    def subtract(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.subtract(x1, x2)

    def multiply(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.multiply(x1, x2)

    def sqrt(self, x: Array) -> Array:
        return self.array_module.sqrt(x)

    def exp(self, x: Array) -> Array:
        return self.array_module.exp(x)

    def negative(self, x: Array) -> Array:
        return self.array_module.negative(x)

    def divide(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.divide(x1, x2)

    def floor_divide(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.floor_divide(x1, x2)

    def minimum(self, x1: Array, x2: Array) -> Array:
        return self.array_module.minimum(x1, x2)

    def maximum(self, x1: Array, x2: Array) -> Array:
        return self.array_module.maximum(x1, x2)

    def power(self, x1: Array, x2: Array) -> Array:
        return self.array_module.power(x1, x2)

    def isnan(self, x: Array) -> Array:
        return self.array_module.isnan(x)

    def logical_or(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.logical_or(x1, x2)

    def logical_and(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.logical_and(x1, x2)

    def logical_not(self, x: Array) -> Array:
        return self.array_module.logical_not(x)

    def greater(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.greater(x1, x2)

    def greater_equal(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.greater_equal(x1, x2)

    def lesser(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.lesser(x1, x2)

    def lesser_equal(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.lesser_equal(x1, x2)

    def equal(self, x1: Array, x2: Array) -> Array:
        x1, x2 = self.normalise_binary_arguments(x1, x2)
        return self.array_module.equal(x1, x2)

    ############################ Utility functions

    def array_equal(self, x1: Array, x2: Array, *, equal_nan: bool = False) -> bool:
        return self.array_module.array_equal(x1, x2, equal_nan=equal_nan)

    ############################ Reducer functions

    def all(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self.array_module.all(x, axis=axis, keepdims=keepdims)

    def any(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self.array_module.any(x, axis=axis, keepdims=keepdims)

    def sum(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: metadata.dtype | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self.array_module.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)

    def prod(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: metadata.dtype | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self.array_module.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)

    def min(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self.array_module.min(x, axis=axis, keepdims=keepdims)

    def max(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return self.array_module.max(x, axis=axis, keepdims=keepdims)

    def argmin(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        return self.array_module.argmin(x, axis=axis, keepdims=keepdims)

    def argmax(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        return self.array_module.argmin(x, axis=axis, keepdims=keepdims)

    ############################ extensions to Array API

    def as_contiguous(self, x: Array) -> Array:
        return self.array_module.ascontiguousarray(x)

    def cumsum(self, x: Array, *, axis: int | None = None) -> Array:
        return self.array_module.cumsum(x, axis=axis)

    def from_buffer(self, buffer, *, dtype=None, count: int = -1) -> Array:
        return self.array_module.frombuffer(buffer, dtype=dtype, count=count)

    def search_sorted(
        self,
        x: Array,
        values: Array,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> Array:
        return self.array_module.searchsorted(x, values, side=side, sorter=sorter)

    def repeat(
        self,
        x: Array,
        repeats: Array | int,
        *,
        axis: int | None = None,
    ) -> Array:
        if isinstance(repeats, Array):
            repeats = repeats

        return self.array_module.repeat(x, repeats, axis=axis)

    def tile(self, x: Array, reps: int) -> Array:
        return self.array_module.tile(x, reps)

    def pack_bits(
        self,
        x: Array,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> Array:
        return self.array_module.packbits(x, axis=axis, bitorder=bitorder)

    def unpack_bits(
        self,
        x: Array,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> Array:
        return self.array_module.unpackbits(
            x, axis=axis, count=count, bitorder=bitorder
        )

    def nan_to_num(
        self,
        x: Array,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> Array:
        return self.array_module.nan_to_num(
            x, copy=copy, nan=nan, posinf=posinf, neginf=neginf
        )

    def is_close(
        self,
        x1: Array,
        x2: Array,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> Array:
        return self.array_module.isclose(
            x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    def count_nonzero(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        return self.array_module.count_nonzero(x, axis=axis, keepdims=keepdims)

    def array_str(
        self,
        x: Array,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        return numpy.array_str(
            x,
            max_line_width=max_line_width,
            precision=precision,
            suppress_small=suppress_small,
        )

    def is_c_contiguous(self, x: Array) -> bool:
        return x.flags["C_CONTIGUOUS"]  # type: ignore

    def to_rectilinear(self, x: Array):
        raise _errors.wrap_error(NotImplementedError)

    def byteswap(self, x: Array, copy: bool = False):
        return x.byteswap(inplace=not copy)  # type: ignore

    def error_state(
        self,
        **kwargs: ErrorStateLiteral,
    ) -> ContextManager:
        raise _errors.wrap_error(NotImplementedError)

    @classmethod
    def is_own_array(cls, x) -> bool:
        """
        Args:
            x: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        raise _errors.wrap_error(NotImplementedError)
