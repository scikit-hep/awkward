# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from abc import abstractmethod

import numpy

from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._singleton import Singleton
from awkward._typing import (
    Any,
    Literal,
    NamedTuple,
    Protocol,
    Self,
    SupportsIndex,
    TypeAlias,
    overload,
)

IndexType: TypeAlias = "int | ArrayLike"


class UniqueAllResult(NamedTuple):
    values: ArrayLike
    indices: ArrayLike
    inverse_indices: ArrayLike
    counts: ArrayLike


class ArrayLike(Protocol):
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
    def shape(self) -> tuple[ShapeItem, ...]:
        ...

    @property
    @abstractmethod
    def size(self) -> ShapeItem:
        ...

    @property
    @abstractmethod
    def nbytes(self) -> ShapeItem:
        ...

    @property
    @abstractmethod
    def T(self) -> Self:
        ...

    @abstractmethod
    def __getitem__(
        self,
        key: SupportsIndex
        | slice
        | Ellipsis
        | tuple[SupportsIndex | slice | Ellipsis | ArrayLike, ...]
        | ArrayLike,
    ) -> Self:
        ...

    @overload
    def __setitem__(
        self,
        key: SupportsIndex
        | slice
        | Ellipsis
        | tuple[SupportsIndex | slice | Ellipsis | ArrayLike, ...]
        | ArrayLike,
        value: int | float | bool | complex | ArrayLike,
    ):
        ...

    @overload
    def __setitem__(
        self,
        key: slice
        | Ellipsis
        | tuple[SupportsIndex | slice | Ellipsis, ...]
        | ArrayLike,
        value: int | float | bool | complex,
    ):
        ...

    @abstractmethod
    def __setitem__(self, key, value):
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
    def __len__(self) -> int:
        ...

    @abstractmethod
    def view(self, dtype: dtype) -> Self:
        ...

    # Scalar UFUNCS
    @abstractmethod
    def __add__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __sub__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __mul__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __truediv__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __floordiv__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __gt__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __lt__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __ge__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __le__(self, other: int | complex | float | Self) -> Self:
        ...

    @abstractmethod
    def __eq__(self, other: int | complex | float | bool | Self) -> Self:
        ...

    @abstractmethod
    def __and__(self, other: int | bool | Self) -> Self:
        ...

    @abstractmethod
    def __or__(self, other: int | bool | Self) -> Self:
        ...

    @abstractmethod
    def __invert__(self) -> Self:
        ...

    def __dlpack_device__(self) -> tuple[int, int]:
        ...

    def __dlpack__(self, stream: Any = None) -> Any:
        ...


class NumpyMetadata(Singleton):
    bool_ = numpy.bool_
    int8 = numpy.int8
    int16 = numpy.int16
    int32 = numpy.int32
    int64 = numpy.int64
    uint8 = numpy.uint8
    uint16 = numpy.uint16
    uint32 = numpy.uint32
    uint64 = numpy.uint64
    longlong = numpy.longlong
    float32 = numpy.float32
    float64 = numpy.float64
    complex64 = numpy.complex64
    complex128 = numpy.complex128
    str_ = numpy.str_
    bytes_ = numpy.bytes_

    intp = numpy.intp
    integer = numpy.integer
    signedinteger = numpy.signedinteger
    unsignedinteger = numpy.unsignedinteger
    floating = numpy.floating
    complexfloating = numpy.complexfloating
    number = numpy.number
    object_ = numpy.object_
    generic = numpy.generic

    dtype = numpy.dtype
    ufunc = numpy.ufunc
    iinfo = numpy.iinfo
    errstate = numpy.errstate
    newaxis = numpy.newaxis

    ndarray = numpy.ndarray

    nan = numpy.nan
    inf = numpy.inf

    nat = numpy.datetime64("NaT")
    datetime_data = numpy.datetime_data

    @property
    def issubdtype(self):
        return numpy.issubdtype

    AxisError = numpy.AxisError


if hasattr(numpy, "float16"):
    NumpyMetadata.float16 = numpy.float16

if hasattr(numpy, "float128"):
    NumpyMetadata.float128 = numpy.float128

if hasattr(numpy, "complex256"):
    NumpyMetadata.complex256 = numpy.complex256

if hasattr(numpy, "datetime64"):
    NumpyMetadata.datetime64 = numpy.datetime64

if hasattr(numpy, "timedelta64"):
    NumpyMetadata.timedelta64 = numpy.timedelta64


class NumpyLike(Singleton, Protocol):
    ############################ Awkward features

    @property
    @abstractmethod
    def supports_structured_dtypes(self) -> bool:
        ...

    @property
    @abstractmethod
    def known_data(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_eager(self) -> bool:
        ...

    ############################ array creation

    @abstractmethod
    def asarray(
        self,
        obj,
        *,
        dtype: numpy.dtype | None = None,
        copy: bool | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def ascontiguousarray(self, x: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def frombuffer(
        self, buffer, *, dtype: numpy.dtype | None = None, count: int = -1
    ) -> ArrayLike:
        ...

    @abstractmethod
    def from_dlpack(self, x: Any) -> ArrayLike:
        ...

    @abstractmethod
    def zeros(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: numpy.dtype | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def ones(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: numpy.dtype | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def empty(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        *,
        dtype: numpy.dtype | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def full(
        self,
        shape: ShapeItem | tuple[ShapeItem, ...],
        fill_value,
        *,
        dtype: numpy.dtype | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def zeros_like(
        self, x: ArrayLike, *, dtype: numpy.dtype | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def ones_like(self, x: ArrayLike, *, dtype: numpy.dtype | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def full_like(
        self, x: ArrayLike, fill_value, *, dtype: numpy.dtype | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def arange(
        self,
        start: float | int,
        stop: float | int | None = None,
        step: float | int = 1,
        *,
        dtype: numpy.dtype | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def meshgrid(
        self, *arrays: ArrayLike, indexing: Literal["xy", "ij"] = "xy"
    ) -> list[ArrayLike]:
        ...

    ############################ testing

    @abstractmethod
    def array_equal(
        self, x1: ArrayLike, x2: ArrayLike, *, equal_nan: bool = False
    ) -> ArrayLike:
        ...

    @abstractmethod
    def searchsorted(
        self,
        x: ArrayLike,
        values: ArrayLike,
        *,
        side: Literal["left", "right"] = "left",
        sorter: ArrayLike | None = None,
    ) -> ArrayLike:
        ...

    ############################ manipulation

    @abstractmethod
    def broadcast_arrays(self, *arrays: ArrayLike) -> list[ArrayLike]:
        ...

    @abstractmethod
    def broadcast_to(self, x: ArrayLike, shape: tuple[ShapeItem, ...]) -> ArrayLike:
        ...

    @overload
    def shape_item_as_index(self, x1: int) -> int:
        ...

    @overload
    def shape_item_as_index(self, x1: type[unknown_length]) -> ArrayLike:
        ...

    @abstractmethod
    def shape_item_as_index(self, x1):
        ...

    @abstractmethod
    def index_as_shape_item(self, x1: IndexType) -> ShapeItem:
        ...

    @abstractmethod
    def derive_slice_for_length(
        self, slice_: slice, length: ShapeItem
    ) -> tuple[IndexType, IndexType, IndexType, ShapeItem]:
        ...

    @abstractmethod
    def regularize_index_for_length(
        self, index: IndexType, length: ShapeItem
    ) -> IndexType:
        ...

    @abstractmethod
    def reshape(
        self, x: ArrayLike, shape: tuple[ShapeItem, ...], *, copy: bool | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def nonzero(self, x: ArrayLike) -> tuple[ArrayLike, ...]:
        ...

    @abstractmethod
    def where(self, condition: ArrayLike, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def unique_values(self, x: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def unique_all(self, x: ArrayLike) -> UniqueAllResult:
        ...

    @abstractmethod
    def sort(
        self,
        x: ArrayLike,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def concat(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int | None = 0,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def repeat(
        self,
        x: ArrayLike,
        repeats: ArrayLike | int,
        *,
        axis: int | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def stack(
        self,
        arrays: list[ArrayLike] | tuple[ArrayLike, ...],
        *,
        axis: int = 0,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def packbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        ...

    @abstractmethod
    def unpackbits(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        count: int | None = None,
        bitorder: Literal["big", "little"] = "big",
    ) -> ArrayLike:
        ...

    @abstractmethod
    def strides(self, x: ArrayLike) -> tuple[ShapeItem, ...]:
        ...

    ############################ ufuncs

    @abstractmethod
    def add(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def logical_or(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def logical_and(
        self, x1: ArrayLike, x2: ArrayLike, *, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def logical_not(
        self, x: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def sqrt(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def exp(self, x: ArrayLike, maybe_out: ArrayLike | None = None) -> ArrayLike:
        ...

    @abstractmethod
    def divide(
        self, x1: ArrayLike, x2: ArrayLike, maybe_out: ArrayLike | None = None
    ) -> ArrayLike:
        ...

    ############################ almost-ufuncs

    @abstractmethod
    def nan_to_num(
        self,
        x: ArrayLike,
        *,
        copy: bool = True,
        nan: int | float | None = 0.0,
        posinf: int | float | None = None,
        neginf: int | float | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def isclose(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def isnan(self, x: ArrayLike) -> ArrayLike:
        ...

    @abstractmethod
    def all(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def any(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def min(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def max(
        self,
        x: ArrayLike,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def count_nonzero(
        self, x: ArrayLike, *, axis: int | None = None, keepdims: bool = False
    ) -> ArrayLike:
        ...

    @abstractmethod
    def cumsum(
        self,
        x: ArrayLike,
        *,
        axis: int | None = None,
        maybe_out: ArrayLike | None = None,
    ) -> ArrayLike:
        ...

    @abstractmethod
    def array_str(
        self,
        x: ArrayLike,
        *,
        max_line_width: int | None = None,
        precision: int | None = None,
        suppress_small: bool | None = None,
    ):
        ...

    @abstractmethod
    def astype(
        self, x: ArrayLike, dtype: numpy.dtype, *, copy: bool | None = True
    ) -> ArrayLike:
        ...

    @abstractmethod
    def can_cast(
        self, from_: numpy.dtype | ArrayLike, to: numpy.dtype | ArrayLike
    ) -> bool:
        ...

    @abstractmethod
    def is_c_contiguous(self, x: ArrayLike) -> bool:
        ...

    @classmethod
    @abstractmethod
    def is_own_array(cls, obj) -> bool:
        ...

    @classmethod
    @abstractmethod
    def is_own_array_type(cls, type_) -> bool:
        ...
