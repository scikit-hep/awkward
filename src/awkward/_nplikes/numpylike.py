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
from typing import Literal, SupportsIndex, SupportsInt, TypeVar, overload

import numpy

import awkward as ak
from awkward._nplikes import dtypes
from awkward._nplikes.dtypes import dtype
from awkward._util import Singleton
from awkward.typing import Protocol, Self, runtime_checkable

ArrayType = TypeVar("ArrayType", bound="Array")
ArrayType_co = TypeVar("ArrayType_co", bound="Array", covariant=True)


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
    def T(self: ArrayType) -> ArrayType:
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
    def __add__(self: ArrayType, other: int | float | complex | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __sub__(self: ArrayType, other: int | float | complex | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __truediv__(
        self: ArrayType, other: int | float | complex | ArrayType
    ) -> ArrayType:
        ...

    @abstractmethod
    def __floordiv__(
        self: ArrayType, other: int | float | complex | ArrayType
    ) -> ArrayType:
        ...

    @abstractmethod
    def __mod__(self: ArrayType, other: int | float | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __mul__(self: ArrayType, other: int | float | complex | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __pow__(self: ArrayType, other: int | float | complex | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __xor__(self: ArrayType, other: int | bool | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __and__(self: ArrayType, other: int | bool | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __or__(self: ArrayType, other: int | bool | ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __lt__(
        self: ArrayType, other: int | float | complex | str | bytes | ArrayType
    ) -> ArrayType:
        ...

    @abstractmethod
    def __le__(
        self: ArrayType, other: int | float | complex | str | bytes | ArrayType
    ) -> ArrayType:
        ...

    @abstractmethod
    def __gt__(
        self: ArrayType, other: int | float | complex | str | bytes | ArrayType
    ) -> ArrayType:
        ...

    @abstractmethod
    def __ge__(
        self: ArrayType, other: int | float | complex | str | bytes | ArrayType
    ) -> ArrayType:
        ...

    @abstractmethod
    def __eq__(self: ArrayType, other: int | float | bool | complex | str | bytes | ArrayType) -> ArrayType:  # type: ignore
        ...

    @abstractmethod
    def __ne__(self: ArrayType, other: int | float | bool | complex | str | bytes | ArrayType) -> ArrayType:  # type: ignore
        ...

    @abstractmethod
    def __abs__(self: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __neg__(self: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __pos__(self: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def __invert__(self: ArrayType) -> ArrayType:
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


@runtime_checkable
class NumpyLike(Protocol[ArrayType], Singleton):

    ############################ array creation

    @abstractmethod
    def asarray(
        self,
        obj,
        *,
        dtype: dtype | None = None,
        copy: bool | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: dtype | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def zeros_like(self, x: ArrayType, *, dtype: dtype | None = None) -> ArrayType:
        ...

    @abstractmethod
    def ones_like(self, x: ArrayType, *, dtype: dtype | None = None) -> ArrayType:
        ...

    @abstractmethod
    def full_like(
        self,
        x: ArrayType,
        fill_value,
        *,
        dtype: dtype | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: dtype | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def meshgrid(
        self, *arrays: ArrayType, indexing: Literal["xy", "ij"] = "xy"
    ) -> ArrayType:
        ...

    ############################ data type functions

    @abstractmethod
    def astype(self, x: ArrayType, dtype: dtype, *, copy: bool = True) -> ArrayType:
        ...

    @abstractmethod
    def broadcast_arrays(self, *arrays: ArrayType) -> list[ArrayType]:
        ...

    # TODO better type signature for unknown shapes
    @abstractmethod
    def broadcast_to(self, x: ArrayType, shape: tuple[SupportsInt, ...]) -> ArrayType:
        ...

    def result_type(self, *arrays_and_dtypes: ArrayType | dtype) -> dtype:
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

    def isdtype(
        self,
        dtype: dtype,
        kind: dtype | str | tuple[dtype | str, ...],
    ) -> bool:
        if isinstance(kind, str):
            if kind == "bool":
                return numpy.issubdtype(dtype, dtypes.bool_)
            elif kind == "signed integer":
                return any([numpy.issubdtype(dtype, c) for c in dtypes._signed_integer])
            elif kind == "unsigned integer":
                return any(
                    [numpy.issubdtype(dtype, c) for c in dtypes._unsigned_integer]
                )
            elif kind == "integral":
                return self.isdtype(dtype, "signed integer") or self.isdtype(
                    dtype, "unsigned integer"
                )
            elif kind == "real floating":
                return any([numpy.issubdtype(dtype, c) for c in dtypes._real_floating])
            elif kind == "complex floating":
                return any(
                    [numpy.issubdtype(dtype, c) for c in dtypes._complex_floating]
                )
            elif kind == "numeric":
                return (
                    self.isdtype(dtype, "integral")
                    or self.isdtype(dtype, "real floating")
                    or self.isdtype(dtype, "complex floating")
                )
            ### Extensions to Array API ###
            elif kind == "timelike":
                return self.isdtype(dtype, dtypes.timedelta64) or self.isdtype(
                    dtype, dtypes.datetime64
                )
            else:
                raise ak._errors.wrap_error(ValueError(f"Invalid kind {kind} given"))
        elif isinstance(kind, tuple):
            return any([self.isdtype(dtype, k) for k in kind])
        else:
            assert isinstance(kind, dtypes.dtype)
            return numpy.issubdtype(dtype, kind)

    def iinfo(self, type: dtype | ArrayType) -> numpy.iinfo:
        return numpy.iinfo(type)

    def finfo(self, type: dtype | ArrayType) -> numpy.finfo:
        return numpy.finfo(type)

    ############################ searching functions

    @abstractmethod
    def nonzero(self, x: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def where(self, condition: ArrayType, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    ############################ set functions

    @abstractmethod
    def unique_counts(self, x: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def unique_values(self, x: ArrayType) -> ArrayType:
        ...

    ############################ manipulation functions

    @abstractmethod
    def concat(
        self, arrays: list[ArrayType] | tuple[ArrayType, ...], *, axis: int | None = 0
    ) -> ArrayType:
        ...

    @abstractmethod
    def stack(
        self, arrays: list[ArrayType] | tuple[ArrayType, ...], *, axis: int = 0
    ) -> ArrayType:
        ...

    ############################ ufuncs

    @abstractmethod
    def add(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def multiply(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def logical_or(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def logical_and(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def logical_not(self, x: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def sqrt(self, x: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def exp(self, x: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def divide(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def equal(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def isnan(self, x: ArrayType) -> ArrayType:
        ...

    ############################ Utility functions

    @abstractmethod
    def all(
        self,
        x: ArrayType,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayType:
        ...

    @abstractmethod
    def any(
        self,
        x: ArrayType,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayType:
        ...

    ############################ Statistical functions

    @abstractmethod
    def sum(
        self,
        x: ArrayType,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtype | None = None,
        keepdims: bool = False,
    ) -> ArrayType:
        ...

    @abstractmethod
    def prod(
        self,
        x: ArrayType,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtype | None = None,
        keepdims: bool = False,
    ) -> ArrayType:
        ...

    @abstractmethod
    def min(
        self,
        x: ArrayType,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayType:
        ...

    @abstractmethod
    def max(
        self,
        x: ArrayType,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> ArrayType:
        ...

    ############################ Searching functions

    @abstractmethod
    def argmin(
        self, x: ArrayType, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayType:
        ...

    @abstractmethod
    def argmax(
        self, x: ArrayType, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayType:
        ...

    ############################ extensions to ArrayType API

    @abstractmethod
    def as_contiguous(self, x: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def cumsum(self, x: ArrayType, *, axis: int | None = None) -> ArrayType:
        ...

    @abstractmethod
    def from_buffer(
        self, buffer: ArrayType, *, dtype: dtype | None = None, count: int = -1
    ) -> ArrayType:  # TODO dtype: float?`
        ...

    @abstractmethod
    def array_equal(
        self, x1: ArrayType, x2: ArrayType, *, equal_nan: bool = False
    ) -> bool:
        ...

    @abstractmethod
    def search_sorted(
        self,
        x: ArrayType,
        values: ArrayType,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> ArrayType:
        ...

    # TODO
    @abstractmethod
    def repeat(self, x: ArrayType, repeats, *, axis=None) -> ArrayType:
        ...

    @abstractmethod
    def tile(self, x: ArrayType, reps: int) -> ArrayType:
        ...

    @abstractmethod
    def pack_bits(
        self,
        x: ArrayType,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayType:
        ...

    @abstractmethod
    def unpack_bits(
        self,
        x: ArrayType,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayType:
        ...

    @abstractmethod
    def minimum(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def maximum(self, x1: ArrayType, x2: ArrayType) -> ArrayType:
        ...

    @abstractmethod
    def nan_to_num(
        self,
        x: ArrayType,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def is_close(
        self,
        x1: ArrayType,
        x2: ArrayType,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> ArrayType:
        ...

    @abstractmethod
    def count_nonzero(
        self, x: ArrayType, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayType:
        ...

    @abstractmethod
    def array_str(
        self,
        x: ArrayType,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ) -> ArrayType:
        ...

    @abstractmethod
    def is_c_contiguous(self, x) -> bool:
        ...

    @abstractmethod
    def to_rectilinear(self, array: ArrayType):
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
