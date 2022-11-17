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
from typing import (
    Iterable,
    Literal,
    Protocol,
    Sized,
    SupportsIndex,
    TypeVar,
    overload,
    runtime_checkable,
)

from awkward._nplikes.dtypes import dtype

NumpyLikeSelf = TypeVar("NumpyLikeSelf", bound="NumpyLike")

ShapeItem = TypeVar("ShapeItem", covariant=True)


class Shape(Protocol, Sized, Iterable[ShapeItem]):
    @overload
    def __getitem__(self, index: SupportsIndex) -> ShapeItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Shape[ShapeItem]:
        ...

    @abstractmethod
    def __getitem__(self, index):
        ...

    def __add__(self, other) -> Shape[ShapeItem]:
        ...


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
    def shape(self) -> Shape:
        ...

    @property
    @abstractmethod
    def size(self) -> ShapeItem:
        ...

    @property
    @abstractmethod
    def T(self) -> Array:
        ...

    @abstractmethod
    def __add__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __sub__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __truediv__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __floordiv__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __mod__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __mul__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __pow__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __xor__(self, other: int | bool | Array) -> Array:
        ...

    @abstractmethod
    def __and__(self, other: int | bool | Array) -> Array:
        ...

    @abstractmethod
    def __or__(self, other: int | bool | Array) -> Array:
        ...

    @abstractmethod
    def __lt__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __le__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __gt__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __ge__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __eq__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __ne__(self, other: int | float | complex | Array) -> Array:
        ...

    @abstractmethod
    def __abs__(self) -> Array:
        ...

    @abstractmethod
    def __neg__(self) -> Array:
        ...

    @abstractmethod
    def __pos__(self) -> Array:
        ...

    @abstractmethod
    def __invert__(self) -> Array:
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


T = TypeVar("T", bound=Array)


@runtime_checkable
class NumpyLike(Protocol[T]):
    known_data: bool
    known_shape: bool
    is_eager: bool

    _instance: NumpyLikeSelf

    @classmethod
    def instance(cls: type[NumpyLikeSelf]) -> NumpyLikeSelf:
        try:
            return cls._instance
        except AttributeError:
            cls._instance = cls()
            return cls._instance

    ############################ array creation

    @abstractmethod
    def asarray(
        self,
        obj,
        *,
        dtype: dtype | None = None,
        copy: bool | None = None,
    ) -> T:
        ...

    @abstractmethod
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype | None = None,
    ) -> T:
        ...

    @abstractmethod
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype | None = None,
    ) -> T:
        ...

    @abstractmethod
    def empty(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: dtype | None = None,
    ) -> T:
        ...

    @abstractmethod
    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value,
        *,
        dtype: dtype | None = None,
    ) -> T:
        ...

    @abstractmethod
    def zeros_like(self, x: T, *, dtype: dtype | None = None) -> T:
        ...

    @abstractmethod
    def ones_like(self, x: T, *, dtype: dtype | None = None) -> T:
        ...

    @abstractmethod
    def full_like(
        self,
        x: T,
        fill_value,
        *,
        dtype: dtype | None = None,
    ) -> T:
        ...

    @abstractmethod
    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: dtype | None = None,
    ) -> T:
        ...

    @abstractmethod
    def meshgrid(self, *arrays: T, indexing: Literal["xy", "ij"] = "xy") -> T:
        ...

    ############################ data type functions

    @abstractmethod
    def astype(self, x: T, dtype: dtype, *, copy: bool = True) -> T:
        ...

    @abstractmethod
    def broadcast_arrays(self, *arrays: T) -> list[T]:
        ...

    # TODO better type signature for unknown shapes
    @abstractmethod
    def broadcast_to(self, x, shape: Shape) -> T:
        ...

    @abstractmethod
    def result_type(self, *arrays_and_dtypes: T | dtype):
        ...

    ############################ searching functions

    @abstractmethod
    def nonzero(self, x: T) -> T:
        ...

    @abstractmethod
    def where(self, condition: T, x1: T, x2: T) -> T:
        ...

    ############################ set functions

    @abstractmethod
    def unique_counts(self, x: T) -> T:
        ...

    @abstractmethod
    def unique_values(self, x: T) -> T:
        ...

    ############################ manipulation functions

    @abstractmethod
    def concat(self, arrays: list[T] | tuple[T, ...], *, axis: int | None = 0) -> T:
        ...

    @abstractmethod
    def stack(self, arrays: list[T] | tuple[T, ...], *, axis: int = 0) -> T:
        ...

    ############################ ufuncs

    @abstractmethod
    def add(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def multiply(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def logical_or(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def logical_and(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def logical_not(self, x: T) -> T:
        ...

    @abstractmethod
    def sqrt(self, x: T) -> T:
        ...

    @abstractmethod
    def exp(self, x: T) -> T:
        ...

    @abstractmethod
    def divide(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def equal(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def isnan(self, x: T) -> T:
        ...

    ############################ Utility functions

    @abstractmethod
    def all(
        self,
        x: T,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> T:
        ...

    @abstractmethod
    def any(
        self,
        x: T,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> T:
        ...

    ############################ Statistical functions

    @abstractmethod
    def sum(
        self,
        x: T,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtype | None = None,
        keepdims: bool = False,
    ) -> T:
        ...

    @abstractmethod
    def prod(
        self,
        x: T,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: dtype | None = None,
        keepdims: bool = False,
    ) -> T:
        ...

    @abstractmethod
    def min(
        self,
        x: T,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> T:
        ...

    @abstractmethod
    def max(
        self,
        x: T,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> T:
        ...

    ############################ Searching functions

    @abstractmethod
    def argmin(self, x: T, *, axis: int | None = None, keepdims: bool = False) -> T:
        ...

    @abstractmethod
    def argmax(self, x: T, *, axis: int | None = None, keepdims: bool = False) -> T:
        ...

    ############################ extensions to T API

    @abstractmethod
    def as_contiguous(self, x: T) -> T:
        ...

    @abstractmethod
    def cumsum(self, x: T, *, axis: int | tuple[int, ...] | None = None) -> T:
        ...

    @abstractmethod
    def from_buffer(
        self, buffer: T, *, dtype: dtype = None
    ) -> T:  # TODO dtype: float?`
        ...

    @abstractmethod
    def array_equal(self, x1: T, x2: T, *, equal_nan: bool = False) -> bool:
        ...

    @abstractmethod
    def search_sorted(
        self,
        x: T,
        values: T,
        *,
        side: Literal["left", "right"] = "left",
        sorter=None,
    ) -> T:
        ...

    # TODO
    @abstractmethod
    def repeat(self, x: T, repeats, *, axis=None) -> T:
        ...

    @abstractmethod
    def tile(self, x: T, reps: int) -> T:
        ...

    @abstractmethod
    def pack_bits(
        self,
        x: T,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> T:
        ...

    @abstractmethod
    def unpack_bits(
        self,
        x: T,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> T:
        ...

    @abstractmethod
    def minimum(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def maximum(self, x1: T, x2: T) -> T:
        ...

    @abstractmethod
    def nan_to_num(
        self,
        x: T,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> T:
        ...

    @abstractmethod
    def is_close(
        self,
        x1: T,
        x2: T,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> T:
        ...

    @abstractmethod
    def count_nonzero(
        self, x: T, *, axis: int | None = None, keepdims: bool = False
    ) -> T:
        ...

    @abstractmethod
    def array_str(
        self,
        x: T,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ) -> T:
        ...

    @abstractmethod
    def is_c_contiguous(self, x) -> bool:
        ...

    @abstractmethod
    def to_rectilinear(self, array: T):
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
