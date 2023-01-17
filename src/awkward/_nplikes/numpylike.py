# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

"""
An Array-API inspired array compatibility layer.

Awkward Array needs to support multiple array backends. The NumpyLike interface is designed
to abstract away the implementation details. Where possible, the public interfaces follow
the Array API standard, making sensible decisions where the standard is unspecified.

The Array API does not yet offer support for complex types, mixed type promotion, or datetimes.
As all of these things are required, the current implementation of NumpyLike is not expected
to replicate the type rules of Array API. Instead, the NumPy promotion rules are used, except
scalars are promoted to 0D arrays, and 0D arrays never use `min_scalar_type`.

We don't implement Device support, as it is not used in Awkward.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ContextManager, Literal, SupportsIndex, SupportsInt, overload

import numpy

import awkward as ak
from awkward._nplikes import metadata
from awkward._nplikes.metadata import dtype
from awkward._singleton import Singleton
from awkward.typing import Protocol, Self, runtime_checkable

ErrorStateLiteral = Literal["ignore", "warn", "raise", "call", "print", "log"]


@runtime_checkable
class Array(Protocol):
    @property
    @abstractmethod
    def dtype(self) -> dtype:
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[SupportsInt, ...]:
        ...

    @property
    @abstractmethod
    def size(self) -> SupportsInt:
        ...

    @property
    @abstractmethod
    def T(self: Array) -> Array:
        ...

    @overload
    def __getitem__(
        self, index: SupportsIndex
    ) -> int | float | complex | str | bytes | bytes:
        ...

    @overload
    def __getitem__(
        self, index: slice | Ellipsis | tuple[SupportsIndex | slice | Ellipsis, ...]
    ) -> Self:
        ...

    @abstractmethod
    def __getitem__(self, index) -> Self:
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        ...

    @abstractmethod
    def __int__(self) -> int:
        ...

    @abstractmethod
    def __index__(self) -> int:
        ...

    @abstractmethod
    def view(self, dtype: dtype) -> Self:
        ...


@runtime_checkable
class NumpyLike(Singleton, Protocol):

    ############################ array creation

    @abstractmethod
    def asarray(
        self,
        obj,
        *,
        dtype: dtype | None = None,
        copy: bool | None = None,
    ) -> Array:
        ...

    @abstractmethod
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype,
    ) -> Array:
        ...

    @abstractmethod
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype,
    ) -> Array:
        ...

    @abstractmethod
    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype,
    ) -> Array:
        ...

    @abstractmethod
    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: dtype,
    ) -> Array:
        ...

    @abstractmethod
    def zeros_like(self, x: Array, *, dtype: dtype | None = None) -> Array:
        ...

    @abstractmethod
    def ones_like(self, x: Array, *, dtype: dtype | None = None) -> Array:
        ...

    @abstractmethod
    def full_like(
        self,
        x: Array,
        fill_value,
        *,
        dtype: dtype | None = None,
    ) -> Array:
        ...

    @abstractmethod
    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: dtype | None = None,
    ) -> Array:
        ...

    @abstractmethod
    def meshgrid(self, *arrays: Array, indexing: Literal["xy", "ij"] = "xy") -> Array:
        ...

    ############################ data type functions

    @abstractmethod
    def astype(self, x: Array, dtype: dtype, *, copy: bool = True) -> Array:
        ...

    @abstractmethod
    def broadcast_arrays(self, *arrays: Array) -> list[Array]:
        ...

    # TODO better type signature for unknown shapes
    @abstractmethod
    def broadcast_to(self, x: Array, shape: tuple[SupportsInt, ...]) -> Array:
        ...

    def result_type(self, *arrays_and_dtypes: Array | dtype) -> dtype:
        all_dtypes: list[dtype] = []
        for item in arrays_and_dtypes:
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                item = item.dtype
            if isinstance(item, dtype):
                all_dtypes.append(item)
            else:
                raise ak._errors.wrap_error(
                    TypeError("result_type() inputs must be array_api arrays or dtypes")
                )

        return numpy.result_type(*all_dtypes)

    def iinfo(self, type: dtype | Array) -> metadata.iinfo:
        if hasattr(type, "dtype"):
            type = type.dtype
        return metadata.iinfo(type)

    def finfo(self, type: dtype | Array) -> metadata.finfo:
        if hasattr(type, "dtype"):
            type = type.dtype
        return metadata.finfo(type)

    ############################ searching functions

    @abstractmethod
    def nonzero(self, x: Array) -> tuple[Array, ...]:
        ...

    @abstractmethod
    def where(self, condition: Array, x1: Array, x2: Array) -> Array:
        ...

    ############################ set functions

    @abstractmethod
    def unique_counts(self, x: Array) -> tuple[Array, Array]:
        ...

    @abstractmethod
    def unique_values(self, x: Array) -> Array:
        ...

    ############################ manipulation functions

    @abstractmethod
    def concat(
        self, arrays: list[Array] | tuple[Array, ...], *, axis: int | None = 0
    ) -> Array:
        ...

    @abstractmethod
    def stack(self, arrays: list[Array] | tuple[Array, ...], *, axis: int = 0) -> Array:
        ...

    ############################ ufuncs

    @abstractmethod
    def add(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def subtract(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def multiply(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def sqrt(self, x: Array) -> Array:
        ...

    @abstractmethod
    def exp(self, x: Array) -> Array:
        ...

    @abstractmethod
    def negative(self, x: Array) -> Array:
        ...

    @abstractmethod
    def divide(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def floor_divide(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def power(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def isnan(self, x: Array) -> Array:
        ...

    @abstractmethod
    def logical_or(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def logical_and(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def logical_not(self, x: Array) -> Array:
        ...

    @abstractmethod
    def greater(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def greater_equal(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def lesser(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def lesser_equal(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def equal(self, x1: Array, x2: Array) -> Array:
        ...

    ############################ Utility functions

    @abstractmethod
    def all(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        ...

    @abstractmethod
    def any(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        ...

    ############################ Statistical functions

    @abstractmethod
    def sum(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtype | None = None,
        keepdims: bool = False,
    ) -> Array:
        ...

    @abstractmethod
    def prod(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtype | None = None,
        keepdims: bool = False,
    ) -> Array:
        ...

    @abstractmethod
    def min(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        ...

    @abstractmethod
    def max(
        self,
        x: Array,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        ...

    ############################ Searching functions

    @abstractmethod
    def argmin(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        ...

    @abstractmethod
    def argmax(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        ...

    ############################ extensions to Array API

    @abstractmethod
    def as_contiguous(self, x: Array) -> Array:
        ...

    @abstractmethod
    def cumsum(self, x: Array, *, axis: int | None = None) -> Array:
        ...

    @abstractmethod
    def from_buffer(
        self, buffer: Array, *, dtype: dtype | None = None, count: int = -1
    ) -> Array:  # TODO dtype: float?`
        ...

    @abstractmethod
    def array_equal(self, x1: Array, x2: Array, *, equal_nan: bool = False) -> bool:
        ...

    @abstractmethod
    def search_sorted(
        self,
        x: Array,
        values: Array,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> Array:
        ...

    # TODO
    @abstractmethod
    def repeat(self, x: Array, repeats, *, axis=None) -> Array:
        ...

    @abstractmethod
    def tile(self, x: Array, reps: int) -> Array:
        ...

    @abstractmethod
    def pack_bits(
        self,
        x: Array,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> Array:
        ...

    @abstractmethod
    def unpack_bits(
        self,
        x: Array,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> Array:
        ...

    @abstractmethod
    def minimum(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def maximum(self, x1: Array, x2: Array) -> Array:
        ...

    @abstractmethod
    def nan_to_num(
        self,
        x: Array,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> Array:
        ...

    @abstractmethod
    def is_close(
        self,
        x1: Array,
        x2: Array,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> Array:
        ...

    @abstractmethod
    def count_nonzero(
        self, x: Array, *, axis: int | None = None, keepdims: bool = False
    ) -> Array:
        ...

    @abstractmethod
    def array_str(
        self,
        x: Array,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ) -> Array:
        ...

    @abstractmethod
    def is_c_contiguous(self, x) -> bool:
        ...

    @abstractmethod
    def to_rectilinear(self, array: Array):
        ...

    @abstractmethod
    def byteswap(self, array: Array, copy: bool = False):
        ...

    @abstractmethod
    def error_state(
        self,
        **kwargs: ErrorStateLiteral,
    ) -> ContextManager:
        ...

    ############################ Awkward features

    @property
    @abstractmethod
    def known_data(self) -> bool:
        ...

    @property
    @abstractmethod
    def known_shape(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_eager(self) -> bool:
        ...

    @classmethod
    @abstractmethod
    def is_own_array(cls, obj) -> bool:
        """
        Args:
            obj: object to test

        Return `True` if the given object is a numpy buffer, otherwise `False`.

        """
        ...
