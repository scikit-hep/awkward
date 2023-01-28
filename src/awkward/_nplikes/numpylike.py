# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

from abc import abstractmethod

import numpy

from awkward._singleton import Singleton
from awkward.typing import (
    Literal,
    Protocol,
    Self,
    SupportsIndex,
    SupportsInt,  # noqa: F401
    TypeAlias,
    overload,
)

ShapeItem: TypeAlias = "SupportsInt | None"


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
    def T(self) -> Self:
        ...

    @abstractmethod
    def __getitem__(  # noqa: F811
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
    def __setitem__(  # noqa: F811
        self,
        key: slice
        | Ellipsis
        | tuple[SupportsIndex | slice | Ellipsis, ...]
        | ArrayLike,
        value: int | float | bool | complex,
    ):
        ...

    @abstractmethod
    def __setitem__(self, key, value):  # noqa: F811
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
    def ascontiguousarray(
        self, x: ArrayLike, *, dtype: numpy.dtype | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def frombuffer(
        self, buffer, *, dtype: numpy.dtype | None = None, count: int = -1
    ) -> ArrayLike:
        ...

    @abstractmethod
    def zeros(
        self, shape: int | tuple[int, ...], *, dtype: numpy.dtype | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def ones(
        self, shape: int | tuple[int, ...], *, dtype: numpy.dtype | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def empty(
        self, shape: int | tuple[int, ...], *, dtype: numpy.dtype | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def full(
        self,
        shape: int | tuple[int, ...],
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
    ) -> bool:
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

    @abstractmethod
    def shape_item_as_scalar(self, x1: ShapeItem):
        ...

    @abstractmethod
    def scalar_as_shape_item(self, x1) -> ShapeItem:
        ...

    @abstractmethod
    def sub_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        ...

    @abstractmethod
    def add_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        ...

    @abstractmethod
    def mul_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        ...

    @abstractmethod
    def div_shape_item(self, x1: ShapeItem, x2: ShapeItem) -> ShapeItem:
        ...

    @abstractmethod
    def reshape(
        self, x: ArrayLike, shape: tuple[int, ...], *, copy: bool | None = None
    ) -> ArrayLike:
        ...

    @abstractmethod
    def nonzero(self, x: ArrayLike) -> tuple[ArrayLike, ...]:
        ...

    @abstractmethod
    def unique_values(self, x: ArrayLike) -> ArrayLike:
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
    def tile(self, x: ArrayLike, reps: int) -> ArrayLike:
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

    @abstractmethod
    def to_rectilinear(self, array: ArrayLike) -> ArrayLike:
        ...

    @classmethod
    @abstractmethod
    def is_own_array(cls, obj) -> bool:
        ...
